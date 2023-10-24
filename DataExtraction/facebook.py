import csv
from facebook_scraper import get_posts

# Open CSV files for writing post details and comments
with open('post_details_nike.csv', 'w', newline='', encoding='utf-8') as f1, open('post_comments_nike.csv', 'w', newline='', encoding='utf-8') as f2:
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)

    # Write headers to CSV files
    writer1.writerow(['Post ID', 'Title', 'Post URL', 'Uploaded Date', 'Likes', 'Comments', 'Shares'])
    writer2.writerow(['Post ID', 'Comment'])

    # Get posts from Adidas' official Facebook account
    for post in get_posts('NintendoAmerica', pages=10):
        # Write post details to CSV file
        writer1.writerow([post['post_id'], post['text'], post['post_url'], post['time'], post['likes'], post['comments'], post['shares']])

        # Get comments from the post
        if post['comments_full'] is not None:
            for comment in post['comments_full']:
                # Write comment to CSV file
                writer2.writerow([post['post_id'], comment['comment_text']])

