# ml-epenthesis

## code/pipeline_learn.py
The following steps are required to run code pipeline_learn.py:
- `git clone https://github.com/YAYAYru/ml_epenthesi`
- `cd ml_epenthesis/code`
- `conda create -n env python=3.7` If no environment
- `source activate env`
- `pip install neptune-client neptune-tensorflow-keras tensorflow`
- `mkdir ../data`
- download two folders skeleton-model and weakly-labeled-movement to the folder ../data
- replace two paths path_folder_csv и path_folder_json в pipeline_learn.py
- `mkdir ../model/` To export a predict model to the folder
- `python pipeline_learn.py`
- See project ml-epenthesis in group signlanguages on site Neptune.ai

When you do not need to export to the site, then replace `OFFLINE_ASYNC="async"` with `OFFLINE_ASYNC="offline"` in code/config.py 

