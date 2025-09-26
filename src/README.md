# 首先需要外网使用earth2studio下载文件到~/.cache/earth2studio文件夹，里面有模型和数据集

# 其次使用 bash.sh安装相应包

# 文件说明

- src
    - training
        -train.py  模型训练文件
    - agent.py  RL+Weather文件
    - config.py 配置文件
    - environment.py  RL环境文件
    - networks.py  RL网络
    - replay_buffer.py  经验池
    
需要将下面code添加到webear.py

```
import aiohttp
from aiohttp_socks import ProxyConnector
proxy_url = "socks5://127.0.0.1:10808"
connector = ProxyConnector.from_url(proxy_url)

# Create an aiohttp ClientSession with the proxy
session = aiohttp.ClientSession(connector=connector)
fs = gcsfs.GCSFileSystem(
    cache_timeout=5,
    token="anon",  # noqa: S106 # nosec B106
    access="read_only",
    block_size=8**20,
    asynchronous=True,
    session_kwargs={"connector": connector}
)
```
