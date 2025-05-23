import pandas as pd 
import re
re_start_symbols = re.compile(r'^[\[\{\(]')
re_end_symbols = re.compile(r'[\]\}\)]$')
together_symbols = re.compile(r'^[\[\{\(](.*)[\]\}\)]$')

# This regex will match strings that start and end with the specified characters
# and do not contain these characters in the middle. 
# pattern = r'^(?:\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\})$'


sub = pd.read_csv('data/subtitles_youtube_sdh.csv', sep=",")

print(sub.shape)
# find items that start and end with symbols
part2 = sub[sub['text'].str.match(together_symbols)]
print(part2.shape)

# remove with longer duration than 10 seconds
part2 = part2[part2['duration'] <= 10]
# remove shorter than 1 second
part2 = part2[part2['duration'] >= 1]
part2 = part2[~part2['text'].str.match(r'^\W*$')]

part2 = part2[~part2['text'].str.contains('\"\"\"')]





print(part2.shape)

part2.to_csv('data/sub3_extracted_startend.csv', index=False)