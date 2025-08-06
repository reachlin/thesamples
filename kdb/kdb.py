# A simple script to download google drive files
# and import into AWS S3 vectors.
# 
# parameters:
# --list: List files in Google Drive
# --save: Save a file to S3 vectors
# --query: Query S3 vectors with a text query
# --file_id: The file ID to download from Google Drive
# --s3_bucket: The S3 bucket name to save vectors
# --s3_index: The S3 index name to save vectors

import argparse
import sys
from google_drive import list_files
from s3_vector import save_file, query_vectors

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description='Knowledge Database (KDB) - Manage documents in S3 Vectors')
    parser.add_argument('--list', action='store_true', help='List files in Google Drive')
    parser.add_argument('--save', action='store_true', help='Save a file to S3 vectors')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI for embedding text-embedding-ada-002(length 1536)')
    parser.add_argument('--query', type=str, help='Query S3 vectors with a text query')
    parser.add_argument('--file_id', type=str, help='The file ID to download from Google Drive')
    parser.add_argument('--s3_bucket', type=str, help='The S3 bucket name to save vectors')
    parser.add_argument('--s3_index', type=str, help='The S3 index name to save vectors')
    
    args = parser.parse_args()
    
    # Check if any action is specified
    if not any([args.list, args.save, args.query]):
        parser.print_help()
        sys.exit(1)
    
    # List files in Google Drive
    if args.list:
        print("Listing files in Google Drive:")
        print("-" * 50)
        try:
            files = list_files()
            for file in files:
                print(f"Name: {file['name']}")
                print(f"ID: {file['id']}")
                print("-" * 50)
        except Exception as e:
            print(f"Error listing files: {e}")
            sys.exit(1)
    
    # Save a file to S3 vectors
    if args.save:
        if not args.file_id or not args.s3_bucket or not args.s3_index:
            print("Error: --save requires --file_id, --s3_bucket, and --s3_index")
            sys.exit(1)
        
        print(f"Saving file {args.file_id} to S3 bucket '{args.s3_bucket}', index '{args.s3_index}'")
        try:
            if args.openai:
                print("Using OpenAI for embedding with text-embedding-ada-002")
                save_file(args.file_id, args.s3_bucket, args.s3_index, embedding='text-embedding-ada-002')
            else:
                save_file(args.file_id, args.s3_bucket, args.s3_index)
            print("File saved successfully!")
        except Exception as e:
            print(f"Error saving file: {e}")
            sys.exit(1)
    
    # Query S3 vectors
    if args.query:
        if not args.s3_bucket or not args.s3_index:
            print("Error: --query requires --s3_bucket and --s3_index")
            sys.exit(1)
        
        print(f"Querying S3 index '{args.s3_index}' in bucket '{args.s3_bucket}'")
        print(f"Query: '{args.query}'")
        print("-" * 50)
        try:
            if args.openai:
                print("Using OpenAI for embedding with text-embedding-ada-002")
                query_vectors(args.s3_bucket, args.s3_index, args.query, embedding='text-embedding-ada-002')
            else:
                query_vectors(args.s3_bucket, args.s3_index, args.query)
        except Exception as e:
            print(f"Error querying vectors: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()