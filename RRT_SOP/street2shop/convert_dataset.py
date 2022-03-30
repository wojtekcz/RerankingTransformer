#%%
from pathlib import Path
import json
import pandas as pd
#%%
s2s_dataset_path = Path('/home/wojciech/Documents/image_retrieval_re-ranking/RerankingTransformer/RRT_SOP/data/street2shop/')
img_json_path = s2s_dataset_path / 'clean_pairs_dresses_v2-wcz3.json'
# %%
img_infos = json.loads(img_json_path.read_text())
# %%
imgs_train_df = pd.DataFrame(img_infos['train'])
imgs_val_df = pd.DataFrame(img_infos['val'])
imgs_test_df = pd.DataFrame(img_infos['test'])
# %%
imgs_train_df
# %%
len(imgs_train_df), len(imgs_val_df), len(imgs_test_df)
# %%
def save_split(split_path: Path, split_df):
    entries = [f'384/street2shop_photos/{r.image},{r.id}' for i,r in split_df.iterrows()]
    split_path.write_text('\n'.join(entries)+'\n')
#%%
save_split(s2s_dataset_path / 'train.txt', imgs_train_df.iloc[0:])
# %%
save_split(s2s_dataset_path / 'valid.txt', imgs_val_df.iloc[0:])
# %%
save_split(s2s_dataset_path / 'test.txt', imgs_test_df.iloc[0:])
# %%
# are all "street" photos first?
imgs_train_df.drop_duplicates('id', keep='first')['type'].value_counts()
imgs_val_df.drop_duplicates('id', keep='first')['type'].value_counts()
imgs_test_df.drop_duplicates('id', keep='first')['type'].value_counts()
# yes :)
# %%
