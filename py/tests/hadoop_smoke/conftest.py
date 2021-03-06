#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
from pyspark.sql import SparkSession
from pysparkling.conf import H2OConf
from pysparkling.context import H2OContext

from tests import unit_test_utils


@pytest.fixture(scope="module")
def spark(spark_conf):
    conf = unit_test_utils.get_default_spark_conf(spark_conf)
    return SparkSession.builder.config(conf=conf).setMaster("yarn-client").getOrCreate()


@pytest.fixture(scope="module")
def hc(spark):
    return H2OContext.getOrCreate(H2OConf().setClusterSize(1))
