import csv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate

api_key = ''

youtube = build('youtube', 'v3', developerKey=api_key)

# Get the channel details
request = youtube.channels().list(
    part='contentDetails,statistics',
    forUsername='adidas'
)
response = request.execute()

# Get the playlist ID for the list of all videos uploaded by the channel
playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

videos = []
next_page_token = None

while True:
    # Get the list of videos in the playlist
    request = youtube.playlistItems().list(
        playlistId=playlist_id,
        part='snippet',
        maxResults=50,
        pageToken=next_page_token
    )
    response = request.execute()

    videos += response['items']

    next_page_token = response.get('nextPageToken')

    if next_page_token is None:
        break

# Open CSV files for writing video details and comments
with open('video_details.csv', 'w', newline='', encoding='utf-8') as f1, open('video_comments.csv', 'w', newline='', encoding='utf-8') as f2:
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)

    # Write headers to CSV files
    writer1.writerow(['Video ID', 'Title', 'Duration', 'Uploaded Date', 'Views', 'Likes', 'Dislikes', 'Comments', 'Shares','Keywords'])
    writer2.writerow(['Video ID', 'Comment'])

    # For each video, get the statistics and comments
    for video in videos:
        video_id = video['snippet']['resourceId']['videoId']
        
        try:
            # Get video details
            request = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            )
            response = request.execute()
            snippet = response['items'][0]['snippet']
            contentDetails = response['items'][0]['contentDetails']
            statistics = response['items'][0]['statistics']
            duration = isodate.parse_duration(contentDetails['duration'])

            keywords = ', '.join(snippet.get('tags', []))  # Get the keywords

            # Write video details to CSV file
            writer1.writerow([video_id, snippet['title'], duration, snippet['publishedAt'], statistics.get('viewCount', 'N/A'), statistics.get('likeCount', 'N/A'), statistics.get('dislikeCount', 'N/A'), statistics.get('commentCount', 'N/A'), statistics.get('favoriteCount', 'N/A'),keywords])

            # Get comments
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100  # Only take top 100 comments for each video due to API quota limit
            )
            response = request.execute()

            # Write comments to CSV file
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                writer2.writerow([video_id, comment])

                # Get replies
                if item['snippet']['totalReplyCount'] > 0:
                    if 'replies' in item.keys():
                        for reply in item['replies']['comments']:
                            reply_comment = reply['snippet']['textDisplay']
                            writer2.writerow([video_id, reply_comment])

        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred for video {video_id}: {e.content}")
