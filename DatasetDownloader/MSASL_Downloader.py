import os
import json
import subprocess
import tqdm
import csv

class MSASLDownloader():
  def __init__(self, save_path = f'./videos'):
    
    self.train_json_path = f'./MSASL_train.json'
    self.val_json_path = f'./MSASL_val.json'
    self.test_json_path = f'./MSASL_test.json'
    
    self.train_json = json.load(open(self.train_json_path))
    self.val_json = json.load(open(self.val_json_path))
    self.test_json = json.load(open(self.test_json_path))
    
    self.classes = json.load(open(f'./MSASL_classes.json'))
    
    self.raw_video_path = f'./raw_videos'
    self.video_save_path = save_path
    
    if not os.path.exists(self.video_save_path):
      os.makedirs(self.video_save_path)
    if not os.path.exists(self.raw_video_path):
      os.makedirs(self.raw_video_path)
    self.logger = open(f'./log.txt', 'w')

  def get_split_list(self, split=100):
    self.split = split
    self.train_split = []
    self.val_split = []
    self.test_split = []
    
    for data in self.train_json:
      if data['text'] in self.classes[:self.split]:
        self.train_split.append(data)
    for data in self.val_json:
      if data['text'] in self.classes[:self.split]:
        self.val_split.append(data)
    for data in self.test_json:
      if data['text'] in self.classes[:self.split]:
        self.test_split.append(data)
    self.split_classes = self.classes[:self.split]
    print(f'Train split: {len(self.train_split)}, Val split: {len(self.val_split)}, Test split: {len(self.test_split)}')
  
  def seconds_to_timestamp(self, seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

  def download_all_data(self):
    self.download_split_data(self.train_json)
    self.download_split_data(self.val_json)
    self.download_split_data(self.test_json)  
  
  def download_a_video(self, raw_name, url):
    if raw_name not in os.listdir(self.raw_video_path):
      download_cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mp4",
        "-o", f"{self.raw_video_path}/{raw_name}",
        url
      ]
      subprocess.run(download_cmd, check=True)

  def clip_a_video(self, raw_name, start_time, end_time, output_path):
    clip_cmd = [
      "ffmpeg",
      "-ss", str(start_time),
      "-to", str(end_time),
      "-i", f"{self.raw_video_path}/{raw_name}",
      "-c", "copy",
      output_path
      ]
    subprocess.run(clip_cmd, check=True)

  def download_clip_a_video(self, data):
    url = data['url']
    text = data['text']
    start_time = data['start_time']
    end_time = data['end_time']
    start = data['start']
    end = data['end']
    label = data['label']
    if text not in self.classes:
      print(f'Class {text} not found in classes.json')
    
    raw_name = f"{url.split('=')[-1]}.mp4"
    save_name = f"{url.split('=')[-1]}_{text}_{label}_{start}_{end}.mp4"
    try:
      self.download_a_video(raw_name, url)
    except Exception as e:
      print(f'Error downloading video {raw_name}: {e}')
      self.logger.write(f'Error downloading video {raw_name}: {e}\n')

    self.clip_a_video(raw_name, self.seconds_to_timestamp(start_time), self.seconds_to_timestamp(end_time), f'{self.video_save_path}/{save_name}')

  def download_split_data(self, json_data):
    for data in tqdm.tqdm(json_data):
      self.download_clip_a_video(data)
      print(f'-*'*50)
	
  def preprocess(self):
    self.label_save_path = f'./MSASL_{self.split}'
    if not os.path.exists(self.label_save_path):
      os.makedirs(self.label_save_path)
    self.train_list = open(f'{self.label_save_path}/train.txt', 'w')
    self.val_list = open(f'{self.label_save_path}/val.txt', 'w')
    self.test_list = open(f'{self.label_save_path}/test.txt', 'w')
    self.label_file = open(f'{self.label_save_path}/labels.csv', 'w', newline='', encoding='gbk')
    self.writer = csv.writer(self.label_file)
    
    self.preprocess_split_data(self.train_split, self.train_list)
    self.preprocess_split_data(self.val_split, self.val_list)
    self.preprocess_split_data(self.test_split, self.test_list)

  def preprocess_split_data(self, json_data, file):
    for data in json_data:
      url = data['url']
      text = data['text']
      start = data['start']
      end = data['end']
      label = data['label']
      save_name = f"{url.split('=')[-1]}_{text}_{label}_{start}_{end}.mp4"
      video_path = f'{self.video_save_path}/{save_name}'
      label_path = f'{label}'
      file.write(f'{video_path} {label_path}\n')
    self.writer.writerow([f'id', f'name'])
    for i, cls_name in enumerate(self.split_classes):
      self.writer.writerow([i, cls_name])

    
  def preprocess_all_split(self):
    self.get_split_list(100)
    self.preprocess()
    self.get_split_list(1000)
    self.preprocess()

  def download_check(self):
    ##### train
    train_video_list = []
    for data in self.train_json:
      url = data['url']
      video_name = f"{url.split('=')[-1]}"
      if video_name not in train_video_list:
        train_video_list.append(f'{video_name}.mp4')
    ##### val
    val_video_list = []
    for data in self.val_json:
      url = data['url']
      video_name = f"{url.split('=')[-1]}"
      if video_name not in val_video_list:
        val_video_list.append(f'{video_name}.mp4')
    ##### test  
    test_video_list = []
    for data in self.test_json:
      url = data['url']
      video_name = f"{url.split('=')[-1]}"
      if video_name not in test_video_list:
        test_video_list.append(f'{video_name}.mp4')
    ##### check
    total_video_list = train_video_list + val_video_list + test_video_list
    total_video_list = list(set(total_video_list))
    print(f'Total videos: {len(total_video_list)}')
    print(f'Train videos: {len(train_video_list)}')
    print(f'Val videos: {len(val_video_list)}')
    print(f'Test videos: {len(test_video_list)}')
    ###
    downloaded_video = os.listdir(self.raw_video_path)
    not_download = list(set(total_video_list).difference(set(downloaded_video)))
    if len(not_download) == 0:
      print('All videos downloaded successfully!')
    else:
      print(f'Not downloaded videos: {len(not_download)}')
      # for video in not_download:
      #   print(video)
        

    
				
if __name__ == f'__main__':
  downloader = MSASLDownloader()
  # downloader.download_all_data()
  # downloader.preprocess_all_split()
  downloader.download_check()

    
