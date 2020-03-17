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

import ai.h2o.sparkling.extensions.serde.ChunkAutoBufferWriter
import ai.h2o.sparkling.frame.H2OChunk
import org.apache.spark.h2o.H2OConf
import org.apache.spark.h2o.utils.NodeDesc
import org.apache.spark.{ml, mllib}

private[backend] class Writer(conf: H2OConf,
                              nodeDesc: NodeDesc,
                              frameName: String,
                              numRows: Int,
                              expectedTypes: Array[Byte],
                              chunkId: Int,
                              maxVecSizes: Array[Int],
                              sparse: Array[Boolean]) extends Closeable {

  private val outputStream = H2OChunk.putChunk(nodeDesc, conf, frameName, numRows, chunkId, expectedTypes, maxVecSizes)
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

  def putNA(sparkIdx: Int): Unit = chunkWriter.writeNA(expectedTypes(sparkIdx))

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
