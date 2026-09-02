"""测试环境不向外部 Langfuse 服务导出 Trace。"""

import os


os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
