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

package ai.h2o.sparkling.local

import ai.h2o.sparkling.{IntegTestHelper, IntegTestStopper, LocalTest}
import ai.h2o.sparkling.examples.ChicagoCrimeAppSmall
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ChicagoCrimeAppSmallSuite extends FunSuite with IntegTestHelper {

  test("Launch Chicago Crime Demo", LocalTest) {
    launch(ChicagoCrimeAppSmall.getClass.getName.replace("$", ""),
      env {
        sparkMaster("local[*]")
        conf("spark.executor.memory", "3g")
        conf("spark.driver.memory", "3g")
      }
    )
  }
}

object ChicagoCrimeAppSmallTest extends IntegTestStopper {
  def main(args: Array[String]): Unit = exitOnException {
    ChicagoCrimeAppSmall.main(args)
  }
}
