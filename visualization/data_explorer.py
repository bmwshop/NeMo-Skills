# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import signal

from callbacks import app
from layouts import get_main_page_layout

signal.signal(signal.SIGALRM, signal.SIG_IGN)


if __name__ == "__main__":
    app.title = "Data Explorer"
    app.layout = get_main_page_layout()
    app.run(
        host='0.0.0.0',
        port='8080',
    )
