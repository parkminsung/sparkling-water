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

import ai.h2o.sparkling.extensions.serde.{ChunkSerdeConstants, SerdeUtils}
import ai.h2o.sparkling.frame.H2OFrame
import ai.h2o.sparkling.utils.ScalaUtils.withResource
import org.apache.spark.h2o.utils.SupportedTypes.Double
import org.apache.spark.h2o.utils.{NodeDesc, ReflectionUtils}
import org.apache.spark.h2o.{H2OContext, _}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.{ExposeUtils, TaskContext, ml, mllib}
import water.{H2O, Key}

import scala.collection.immutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._


private[backend] object Converter {
  type SparkJob[T] = (TaskContext, Iterator[T]) => (Int, Long)
  type UploadPlan = immutable.Map[Int, NodeDesc]

  /**
   * Converts Spark DataFrame to H2O Frame using specified conversion function
   *
   * @param hc            H2O context
   * @param df            Data frame to convert
   * @return H2OFrame Key
   */
  def convert[T: ClassTag : TypeTag](hc: H2OContext, df: DataFrame, frameKeyName: Option[String]): String = {
    import ai.h2o.sparkling.ml.utils.SchemaUtils._

    val flatDataFrame = flattenDataFrame(df)
    val dfRdd = flatDataFrame.rdd
    val keyName = frameKeyName.getOrElse("frame_rdd_" + dfRdd.id + Key.rand())

    val elemMaxSizes = collectMaxElementSizes(flatDataFrame)
    val elemStartIndices = collectElemStartPositions(elemMaxSizes)
    val vecIndices = collectVectorLikeTypes(flatDataFrame.schema).toArray
    val sparseInfo = collectSparseInfo(flatDataFrame, elemMaxSizes)
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
    // prepare required metadata
    val uploadPlan = scheduleUpload(nonEmptyPartitions.size, rdd)
    val operation: SparkJob[Row] = perDataFramePartition(conf, elemStartIndices, maxVecSizes, keyName, expectedTypes,
      uploadPlan, sparseInfo, nonEmptyPartitions, partitionSizes)
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
      Converter.internalJavaClassOf(f.dataType)
    }.toArray
    prepareExpectedTypes(internalJavaClasses)
  }

  private[backend] def prepareExpectedTypes(classes: Array[Class[_]]): Array[Byte] = {
    classes.map { clazz =>
      if (clazz == classOf[java.lang.Boolean]) {
        ChunkSerdeConstants.EXPECTED_BOOL
      } else if (clazz == classOf[java.lang.Byte]) {
        ChunkSerdeConstants.EXPECTED_BYTE
      } else if (clazz == classOf[java.lang.Short]) {
        ChunkSerdeConstants.EXPECTED_SHORT
      } else if (clazz == classOf[java.lang.Character]) {
        ChunkSerdeConstants.EXPECTED_CHAR
      } else if (clazz == classOf[java.lang.Integer]) {
        ChunkSerdeConstants.EXPECTED_INT
      } else if (clazz == classOf[java.lang.Long]) {
        ChunkSerdeConstants.EXPECTED_LONG
      } else if (clazz == classOf[java.lang.Float]) {
        ChunkSerdeConstants.EXPECTED_FLOAT
      } else if (clazz == classOf[java.lang.Double]) {
        ChunkSerdeConstants.EXPECTED_DOUBLE
      } else if (clazz == classOf[java.lang.String]) {
        ChunkSerdeConstants.EXPECTED_STRING
      } else if (clazz == classOf[java.sql.Timestamp] || clazz == classOf[java.sql.Date]) {
        ChunkSerdeConstants.EXPECTED_TIMESTAMP
      } else if (clazz == classOf[org.apache.spark.ml.linalg.Vector]) {
        ChunkSerdeConstants.EXPECTED_VECTOR
      } else {
        throw new RuntimeException("Unsupported class: " + clazz)
      }
    }
  }

  private def perDataFramePartition(conf: H2OConf,
                                    elemStartIndices: Array[Int],
                                    maxVecSizes: Array[Int],
                                    keyName: String,
                                    expectedTypes: Array[Byte],
                                    uploadPlan: Converter.UploadPlan,
                                    sparse: Array[Boolean],
                                    partitions: Seq[Int],
                                    partitionSizes: Map[Int, Int])(context: TaskContext, it: Iterator[Row]): (Int, Long) = {
    val chunkIdx = partitions.indexOf(context.partitionId())
    val partitionSize = partitionSizes(context.partitionId())
    withResource(
      new Writer(conf,
        uploadPlan(chunkIdx),
        keyName,
        partitionSize,
        expectedTypes,
        chunkIdx,
        maxVecSizes,
        sparse)) { writer =>
      it.foldLeft(0) {
        case (localRowIdx, row) => sparkRowToH2ORow(row, localRowIdx, writer, elemStartIndices, maxVecSizes)
      }
    }
    (chunkIdx, partitionSize)
  }

  /**
   * Converts a single Spark Row to H2O Row with expanded vectors and arrays
   */
  private def sparkRowToH2ORow(row: Row, rowIdx: Int, con: Writer, elemStartIndices: Array[Int], maxVecSizes: Array[Int]): Int = {
    con.startRow(rowIdx)
    row.schema.fields.zipWithIndex.foreach { case (entry, idxField) =>
      val idxH2O = elemStartIndices(idxField)
      if (row.isNullAt(idxField)) {
        con.putNA(idxH2O, idxField)
      } else {
        entry.dataType match {
          case BooleanType => con.put(idxH2O, row.getBoolean(idxField))
          case ByteType => con.put(idxH2O, row.getByte(idxField))
          case ShortType => con.put(idxH2O, row.getShort(idxField))
          case IntegerType => con.put(idxH2O, row.getInt(idxField))
          case LongType => con.put(idxH2O, row.getLong(idxField))
          case FloatType => con.put(idxH2O, row.getFloat(idxField))
          case _: DecimalType => con.put(idxH2O, row.getDecimal(idxField).doubleValue())
          case DoubleType => con.put(idxH2O, row.getDouble(idxField))
          case StringType => con.put(idxH2O, row.getString(idxField))
          case TimestampType => con.put(idxH2O, row.getAs[java.sql.Timestamp](idxField))
          case DateType => con.put(idxH2O, row.getAs[java.sql.Date](idxField))
          case v if ExposeUtils.isMLVectorUDT(v) => con.putVector(idxH2O, row.getAs[ml.linalg.Vector](idxField), maxVecSizes(idxField))
          case _: mllib.linalg.VectorUDT => con.putVector(idxH2O, row.getAs[mllib.linalg.Vector](idxField), maxVecSizes(idxField))
          case udt if ExposeUtils.isUDT(udt) => throw new UnsupportedOperationException(s"User defined type is not supported: ${udt.getClass}")
          case unsupported => throw new UnsupportedOperationException(s"Data of type ${unsupported.getClass} are not supported for the conversion" +
            s"to H2OFrame.")
        }
      }
    }
    rowIdx + 1
  }

  private def scheduleUpload[T](numPartitions: Int, rdd: H2OAwareRDD[T]): UploadPlan = {
    val hc = H2OContext.ensure("H2OContext needs to be running")
    val nodes = hc.getH2ONodes()
    if (hc.getConf.runsInInternalClusterMode) {
      rdd.mapPartitionsWithIndex { case(idx, _) =>
        Iterator.single((idx, NodeDesc(H2O.SELF)))
      }.collect().toMap
    } else {
      val uploadPlan = (0 until numPartitions).zip(Stream.continually(nodes).flatten).toMap
      uploadPlan
    }
  }

  def internalJavaClassOf(dt: DataType): Class[_] = {
    dt match {
      case n if n.isInstanceOf[DecimalType] & n.getClass.getSuperclass != classOf[DecimalType] => Double.javaClass
      case v if ExposeUtils.isAnyVectorUDT(v) => classOf[Vector]
      case _: DataType => ReflectionUtils.supportedTypeOf(dt).javaClass
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
