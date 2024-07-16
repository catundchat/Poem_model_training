# Poem model training
基于多个开源大模型创建诗歌模型以撰写五言绝句，五言律诗，七言绝句，七言律诗，清平乐(双调46字)，忆江南(双调54字)

## Data Preparation 

[Chinese poetry](https://github.com/chinese-poetry/chinese-poetry): Traditional Chinese (need to convert traditional Chinese into simplified Chinese)

[Poetry](https://github.com/Werneror/Poetry): Simplified Chinese

Dataset format: We choose poems with a title of 2-6 hanzis to construct the dataset, json and jsonl are permitted, then trimmed it into following format about 20000 poems: {"query": "11111", "response": "22222"}
