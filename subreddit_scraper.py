import requests
import json
import os

# Thousands of posts to scrape
n = 100
# Subreddits to scrape
subreddits = [
    'learnpython',
    'cpp',
    'Republican',
    'Democrat',
    'mac',
    'windows',
    'askmen',
    'askwomen',
    'redpill',
    'bluepill',
    'pokemon',
    'digimon',
    'funny',
    'sad',
    'stanford',
    'berkeley',
    'bitcoin',
    'ethereum',
    'ps3',
    'xbox',
    'android',
    'ios',
    'nike',
    'adidas',
    'prolife',
    'prochoice'
    ]
# Where to store data
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

url_template = "https://api.pushshift.io/reddit/search/submission/?subreddit=%s&sort=desc&sort_type=created_utc&after=%s&size=1000"

for i, subreddit in enumerate(subreddits):
  # Reset scrape start date
  start_date = 1523588521
  all_results = []
  try:
    for j in range(0, n):
      if j % 10 == 0:
        print("On index %d of %d for subreddit %s (%d/%d)." % (j + 1, n, subreddit, i + 1, len(subreddits)))

      data = json.loads(requests.get(url_template % (subreddit, start_date)).content)['data']
      if len(data):
        start_date = data[-1]['created_utc']
        all_results.extend(data)

    json.dump(all_results, open("data/results_%s.json" % subreddit, 'w'))
  except Exception as e:
    print('Error :(((((((')
    print(e)
