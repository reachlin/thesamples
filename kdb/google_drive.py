from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import io
import os

def _get_google_drive_service():
    # Authenticate
    SCOPES = [ 'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        #'https://www.googleapis.com/auth/drive',
    ]

    SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_DRIVE_SVC_TOKEN_FILE", 'google_drive_service_account.json')

    # Service account credentials with user impersonation
    credentials = ServiceAccountCredentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=SCOPES,
        subject=os.getenv("GOOGLE_DRIVE_OWNER_EMAIL", "me")  # The email of the user to impersonate
    )

    drive_service = build('drive', 'v3', credentials=credentials)
    print(drive_service.about())
    return drive_service

def list_files():
    """List all files from Google Drive.
    
    Args:
        drive_service: Authenticated Google Drive service instance.
        
    Returns:
        List of file metadata dictionaries containing 'id' and 'name'.
    """
    results = []
    try:
        drive_service = _get_google_drive_service()
        # First, try to list all files (this will help diagnose permission issues)
        print("Attempting to list files from Google Drive...")
        
        page_token = None
        while True:
            try:
                # List all files the user has access to
                response = drive_service.files().list(
                    q="trashed = false",  # Only show non-trashed files
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=100,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()
                
                files = response.get('files', [])
                print(f"Found {len(files)} files in this page")
                results.extend(files)
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                    
            except Exception as e:
                print(f"Error fetching page: {str(e)}")
                break
                
    except Exception as error:
        print(f"Fatal error in list_files: {error}")
        print("Make sure you have the correct Drive API scopes and permissions.")
        print("Required scopes: https://www.googleapis.com/auth/drive.readonly")
        
    print(f"Total files found: {len(results)}")
    return results

def download_file(file_id):
    try:
        drive_service = _get_google_drive_service()
        # First get the file metadata to check the MIME type
        file_metadata = drive_service.files().get(
            fileId=file_id,
            fields='mimeType,exportLinks,name'
        ).execute()
        
        file_name = file_metadata.get('name', '')
        mime_type = file_metadata.get('mimeType', '')
        
        # If it's a Google Workspace file, we need to export it
        if 'google-apps.' in mime_type:
            # Default to markdown for text-based documents
            if 'document' in mime_type or 'text' in mime_type:
                export_mime = 'text/markdown'
                if not file_name.endswith('.md'):
                    file_name += '.md'
            # Handle spreadsheets as CSV
            elif 'spreadsheet' in mime_type:
                export_mime = 'text/csv'
                if not file_name.endswith('.csv'):
                    file_name += '.csv'
            # Handle presentations as PDF (better for markdown conversion)
            elif 'presentation' in mime_type:
                export_mime = 'application/pdf'
                if not file_name.endswith('.pdf'):
                    file_name += '.pdf'
            # For other Google Workspace files, try to export as text
            else:
                export_mime = 'text/plain'
                if not file_name.endswith('.txt'):
                    file_name += '.txt'
            
            request = drive_service.files().export_media(
                fileId=file_id,
                mimeType=export_mime
            )
        else:
            # For regular files, download directly
            request = drive_service.files().get_media(fileId=file_id)
        
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")
        file.seek(0)
        return file_name, file
    except Exception as error:
        print(f"Error: {error}")
        raise  # Re-raise the exception to fail the test

def test_list_files():
    files = list_files()
    print(files)
    assert len(files) > 0

def test_download_file():
    files = list_files()
    file_id = files[0]['id']
    file_name, file = download_file(file_id)
    print(f"file name: {file_name}")
    # save file to disk
    with open(file_name, 'wb') as f:
        f.write(file.read())
    # check local file size
    local_file_size = os.path.getsize(file_name)
    assert local_file_size > 0  

if __name__ == "__main__":
    test_list_files()
    test_download_file()