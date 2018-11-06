#https://api.pushshift.io/reddit/search/submission/?subreddit=learnpython&sort=desc&sort_type=created_utc&after=1523588521&size=100000
import requests
import json
start_date = 1523588521
subreddit = 'learnpython'
url = "https://api.pushshift.io/reddit/search/submission/?subreddit=%s&sort=desc&sort_type=created_utc&after=%s&size=1000"

all_results = []

for i in range(0, 100):
    print("On index %s of %s" % (i, i))
    data = json.loads(requests.get(url % (subreddit, start_date)).content)['data']
    start_date = data[-1]['created_utc']
    all_results.extend(data)

json.dumps("results.json", all_results)
print(len(all_results))