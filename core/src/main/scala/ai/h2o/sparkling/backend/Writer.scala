/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package ai.h2o.sparkling.backend

import java.io.Closeable

import ai.h2o.sparkling.backend.utils.ConversionUtils.H2OTypesFromClasses
import ai.h2o.sparkling.extensions.serde.{ChunkAutoBufferWriter, SerdeUtils}
import ai.h2o.sparkling.frame.{H2OChunk, H2OFrame}
import ai.h2o.sparkling.utils.ScalaUtils.withResource
import org.apache.spark.h2o.utils.SupportedTypes.Double
import org.apache.spark.h2o.utils.{NodeDesc, ReflectionUtils}
import org.apache.spark.h2o.{H2OContext, RDD}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.{ExposeUtils, TaskContext, ml, mllib}
import water.{H2O, Key}

import scala.collection.immutable

private[backend] class Writer(nodeDesc: NodeDesc,
                              metadata: WriterMetadata,
                              numRows: Int,
                              chunkId: Int) extends Closeable {

  private val outputStream = H2OChunk.putChunk(nodeDesc,
    metadata.conf,
    metadata.frameId,
    numRows,
    chunkId,
    metadata.H2OTypes,
    metadata.maxVectorSizes)

  private val chunkWriter = new ChunkAutoBufferWriter(outputStream)

  def put(data: Boolean): Unit = chunkWriter.writeBoolean(data)

  def put(data: Byte): Unit = chunkWriter.writeByte(data)

  def put(data: Char): Unit = chunkWriter.writeChar(data)

  def put(data: Short): Unit = chunkWriter.writeShort(data)

  def put(data: Int): Unit = chunkWriter.writeInt(data)

  def put(data: Long): Unit = chunkWriter.writeLong(data)

  def put(data: Float): Unit = chunkWriter.writeFloat(data)

  def put(data: Double): Unit = chunkWriter.writeDouble(data)

  def put(data: java.sql.Timestamp): Unit = chunkWriter.writeTimestamp(data)

  def put(data: java.sql.Date): Unit = chunkWriter.writeLong(data.getTime)

  def put(data: String): Unit = chunkWriter.writeString(data)

  def putNA(sparkIdx: Int): Unit = chunkWriter.writeNA(metadata.H2OTypes(sparkIdx))

  def putSparseVector(vector: ml.linalg.SparseVector): Unit = chunkWriter.writeSparseVector(vector.indices, vector.values)

  def putDenseVector(vector: ml.linalg.DenseVector): Unit = chunkWriter.writeDenseVector(vector.values)

  def putVector(vector: mllib.linalg.Vector): Unit = putVector(vector.asML)

  def putVector(vector: ml.linalg.Vector): Unit = {
    vector match {
      case sparseVector: ml.linalg.SparseVector =>
        putSparseVector(sparseVector)
      case denseVector: ml.linalg.DenseVector =>
        putDenseVector(denseVector)
    }
  }

  def close(): Unit = chunkWriter.close()
}

object Writer {

  type SparkJob[T] = (TaskContext, Iterator[T]) => (Int, Long)
  type UploadPlan = immutable.Map[Int, NodeDesc]

  /**
   * Converts Spark DataFrame to H2O Frame using specified conversion function
   *
   * @param hc H2O context
   * @param df Data frame to convert
   * @return H2OFrame Key
   */
  def convert(hc: H2OContext, df: DataFrame, frameKeyName: Option[String]): String = {
    import ai.h2o.sparkling.ml.utils.SchemaUtils._

    val flatDataFrame = flattenDataFrame(df)
    val dfRdd = flatDataFrame.rdd
    val keyName = frameKeyName.getOrElse("frame_rdd_" + dfRdd.id + Key.rand())

    val elemMaxSizes = collectMaxElementSizes(flatDataFrame)
    val vecIndices = collectVectorLikeTypes(flatDataFrame.schema).toArray
    // Expands RDD's schema ( Arrays and Vectors)
    val flatRddSchema = expandedSchema(flatDataFrame.schema, elemMaxSizes)
    // Patch the flat schema based on information about types
    val fnames = flatRddSchema.map(_.name).toArray
    val maxVecSizes = vecIndices.map(elemMaxSizes(_))
    val expectedTypes = determineExpectedTypes(flatDataFrame)

    val conf = hc.getConf
    H2OFrame.initializeFrame(conf, keyName, fnames)

    val rdd = new H2OAwareRDD(hc.getH2ONodes(), dfRdd)
    val partitionSizes = getNonEmptyPartitionSizes(rdd)
    val nonEmptyPartitions = getNonEmptyPartitions(partitionSizes)

    val uploadPlan = scheduleUpload(nonEmptyPartitions.size, rdd)
    val metadata = WriterMetadata(conf, keyName, expectedTypes, maxVecSizes)
    val operation: SparkJob[Row] = perDataFramePartition(metadata, uploadPlan, nonEmptyPartitions, partitionSizes)
    val rows = hc.sparkContext.runJob(rdd, operation, nonEmptyPartitions) // eager, not lazy, evaluation
    val res = new Array[Long](nonEmptyPartitions.size)
    rows.foreach { case (cidx, nrows) => res(cidx) = nrows }
    // get the vector types from expected types
    val types = SerdeUtils.expectedTypesToVecTypes(expectedTypes, maxVecSizes)
    H2OFrame.finalizeFrame(conf, keyName, res, types)
    keyName
  }

  private def determineExpectedTypes(flatDataFrame: DataFrame): Array[Byte] = {
    val internalJavaClasses = flatDataFrame.schema.map { f =>
      f.dataType match {
        case n if n.isInstanceOf[DecimalType] & n.getClass.getSuperclass != classOf[DecimalType] => Double.javaClass
        case v if ExposeUtils.isAnyVectorUDT(v) => classOf[Vector]
        case dt: DataType => ReflectionUtils.supportedTypeOf(dt).javaClass
      }
    }.toArray
    H2OTypesFromClasses(internalJavaClasses)
  }

  private def perDataFramePartition(metadata: WriterMetadata,
                                    uploadPlan: UploadPlan,
                                    partitions: Seq[Int],
                                    partitionSizes: Map[Int, Int])(context: TaskContext, it: Iterator[Row]): (Int, Long) = {
    val chunkIdx = partitions.indexOf(context.partitionId())
    val numRows = partitionSizes(context.partitionId())
    withResource(new Writer(uploadPlan(chunkIdx), metadata, numRows, chunkIdx)) { writer =>
      it.foreach { row => sparkRowToH2ORow(row, writer) }
    }
    (chunkIdx, numRows)
  }

  private def sparkRowToH2ORow(row: Row, con: Writer): Unit = {
    row.schema.fields.zipWithIndex.foreach { case (entry, idxField) =>
      if (row.isNullAt(idxField)) {
        con.putNA(idxField)
      } else {
        entry.dataType match {
          case BooleanType => con.put(row.getBoolean(idxField))
          case ByteType => con.put(row.getByte(idxField))
          case ShortType => con.put(row.getShort(idxField))
          case IntegerType => con.put(row.getInt(idxField))
          case LongType => con.put(row.getLong(idxField))
          case FloatType => con.put(row.getFloat(idxField))
          case _: DecimalType => con.put(row.getDecimal(idxField).doubleValue())
          case DoubleType => con.put(row.getDouble(idxField))
          case StringType => con.put(row.getString(idxField))
          case TimestampType => con.put(row.getAs[java.sql.Timestamp](idxField))
          case DateType => con.put(row.getAs[java.sql.Date](idxField))
          case v if ExposeUtils.isMLVectorUDT(v) => con.putVector(row.getAs[ml.linalg.Vector](idxField))
          case _: mllib.linalg.VectorUDT => con.putVector(row.getAs[mllib.linalg.Vector](idxField))
          case udt if ExposeUtils.isUDT(udt) => throw new UnsupportedOperationException(s"User defined type is not supported: ${udt.getClass}")
          case unsupported => throw new UnsupportedOperationException(s"Data of type ${unsupported.getClass} are not supported for the conversion" +
            s"to H2OFrame.")
        }
      }
    }
  }

  private def scheduleUpload[T](numPartitions: Int, rdd: H2OAwareRDD[T]): UploadPlan = {
    val hc = H2OContext.ensure("H2OContext needs to be running")
    val nodes = hc.getH2ONodes()
    if (hc.getConf.runsInInternalClusterMode) {
      rdd.mapPartitionsWithIndex { case (idx, _) =>
        Iterator.single((idx, NodeDesc(H2O.SELF)))
      }.collect().toMap
    } else {
      val uploadPlan = (0 until numPartitions).zip(Stream.continually(nodes).flatten).toMap
      uploadPlan
    }
  }

  private def getNonEmptyPartitionSizes[T](rdd: RDD[T]): Map[Int, Int] = {
    rdd.mapPartitionsWithIndex {
      case (idx, it) => if (it.nonEmpty) {
        Iterator.single((idx, it.size))
      } else {
        Iterator.empty
      }
    }.collect().toMap
  }

  private def getNonEmptyPartitions(partitionSizes: Map[Int, Int]): Seq[Int] = {
    partitionSizes.keys.toSeq.sorted
  }
}