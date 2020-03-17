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

import ai.h2o.sparkling.frame.{H2OChunk, H2OFrame}
import org.apache.spark.Partition

/**
 * Contains functions that are shared between H2O DataFrames and RDDs.
 */
private[backend] trait H2OSparkEntity {
  /** Underlying H2O Frame */
  val frame: H2OFrame

  /** Cache frame key to get H2OFrame from the H2O backend */
  val frameKeyName: String = frame.frameId

  /** Number of chunks per a vector */
  val numChunks: Int = frame.chunks.length

  /** Create new types list which describes expected types in a way external H2O backend can use it. This list
   * contains types in a format same for H2ODataFrame and H2ORDD */
  val expectedTypes: Array[Byte]

  /** Chunk locations helps us to determine the node which really has the data we needs. */
  val chksLocation: Option[Array[H2OChunk]] = Some(frame.chunks)

  /** Selected column indices */
  val selectedColumnIndices: Array[Int]

  protected def getPartitions: Array[Partition] = {
    val res = new Array[Partition](numChunks)
    for (i <- 0 until numChunks) res(i) = new Partition {
      val index: Int = i
    }
    res
  }

  /** Base implementation for iterator over rows stored in chunks for given partition. */
  trait H2OChunkIterator[+A] extends Iterator[A] {

    val reader: Reader

    override def hasNext: Boolean = reader.hasNext
  }

}
