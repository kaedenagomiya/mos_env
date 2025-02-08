
## install

if use uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
from [installing uv for manage python](https://docs.astral.sh/uv/getting-started/installation/)

```
git clone https://github.com/kaedenagomiya/mos_env.git
cd mos_env
uv sync
```

## run

```
uv run streamlit run  app_mos.py --server.port 8051
```