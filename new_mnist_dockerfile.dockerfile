# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile
FROM python:3.8-slim
WORKDIR /root

COPY requirements.txt /root/requirements.txt
COPY setup.py /root/setup.py
COPY src /root/src
COPY data /root/data


RUN pip install -r /root/requirements.txt --no-cache-dir
RUN pip install -e /root/.

RUN wandb login 5126bc75ebb9dbf628dc84fa81ad8be26d701991 # Private account API key

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "src/models/train_model.py"]
