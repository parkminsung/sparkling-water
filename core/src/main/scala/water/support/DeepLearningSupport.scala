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
package water.support

import hex.deeplearning.{DeepLearning, DeepLearningModel}
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import org.apache.spark.h2o._

/**
  * Support class to create and train Deep Learning Model
  */
trait DeepLearningSupport {

  /**
    * Create Deep Learning Model for the basic usage. This method exposes
    * the basic configuration of the model. If you need to specify some arguments which are not exposed, please
    * use the DeepLearning model via Sparkling Water pipelines API or using the raw java API
    * @param train frame to train
    * @param valid validation frame
    * @param response response column
    * @param modelId name of the model
    * @param epochs number of epochs
    * @param l1 l1 regularization
    * @param l2 l2 regularization
    * @param activation activation function
    * @param hidden number and size of the hidden layers
    * @tparam T H2O Frame Type
    * @return Deep Learning Model
    */
  def DLModel[T <: Frame](train: T, valid: T, response: String,
              modelId: String = "model",
              epochs: Int = 10, l1: Double = 0.0001, l2: Double = 0.0001,
              activation: Activation = Activation.RectifierWithDropout,
              hidden: Array[Int] = Array(200, 200)): DeepLearningModel = {

    val dlParams = new DeepLearningParameters()
    dlParams._train = train._key
    dlParams._valid = if (valid != null) {
      valid._key
    } else {
      null
    }
    dlParams._response_column = response
    dlParams._epochs = epochs
    dlParams._l1 = l1
    dlParams._l2 = l2
    dlParams._activation = activation
    dlParams._hidden = hidden

    // Create a job
    val dl = new DeepLearning(dlParams, water.Key.make(modelId))
    val model = dl.trainModel.get
    model
  }
}

// Create companion object
object DeepLearningSupport extends DeepLearningSupport
