# Google Drive Authentication Setup

This document explains how to set up Google Drive authentication for the Kitview Desktop chatbot project and generate the required `token.json` file.

## Overview

The project uses Google Drive API to automatically download knowledge base files. Authentication is handled through OAuth 2.0 flow, which generates access and refresh tokens stored in `token.json`.

## Prerequisites

### 1. Google Cloud Console Setup

1. **Create or Select a Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select an existing one
   - Note the project ID for reference

2. **Enable Google Drive API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"

3. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application" as application type
   - Name your OAuth client (e.g., "Kitview Desktop")
   - Download the JSON file (rename to `client_secrets.json`)

### 2. Project Configuration

1. **Place Configuration Files**:
   ```
   config/
   ├── client_secrets.json    # OAuth credentials from Google Cloud Console
   └── credentials.json       # Auto-generated after first auth (optional)
   ```

2. **Environment Variables**:
   Create/update your `.env` file:
   ```env
   CLIENT_SECRETS=config/client_secrets.json
   GOOGLE_DRIVE_TARGET_FOLDER_ID=your_folder_id_here
   ```

3. **Verify settings.yaml**:
   Ensure your `settings.yaml` contains:
   ```yaml
   client_config_file: config/client_secrets.json
   save_credentials: True
   save_credentials_backend: file
   save_credentials_file: config/credentials.json
   get_refresh_token: True
   oauth_scope:
     - https://www.googleapis.com/auth/drive.readonly
   redirect_uri: http://127.0.0.1:8765/
   ```

## Authentication Methods

### Method 1: Direct Script Execution (Recommended)

Run the Google helper script directly:

```bash
cd "path/to/your/project"
python Helpers/google.py
```

This will:
- Start the OAuth flow automatically
- Open your browser for authentication
- Generate `token.json` in the `Helpers/` directory

### Method 2: Programmatic Authentication

```python
from Helpers.google import get_drive

# This triggers authentication if no valid tokens exist
drive = get_drive()
```

### Method 3: Through Download Function

```python
from Helpers.google import download_knowledge_files_from_googleDrive

# This will authenticate and download files
download_knowledge_files_from_googleDrive()
```

## Authentication Flow

### First-Time Setup

1. **Script Execution**: Run one of the methods above
2. **Browser Opens**: Automatic redirect to Google OAuth consent screen
3. **User Login**: Sign in with your Google account
4. **Grant Permissions**: Approve Drive access for your application
5. **Redirect**: Browser redirects to `http://127.0.0.1:8765/`
6. **Token Generation**: `token.json` is automatically created

### Subsequent Usage

- **Automatic**: Existing tokens are loaded from `token.json`
- **Refresh**: Expired access tokens are automatically refreshed
- **No Browser**: No user interaction required for future runs

## Generated token.json Structure

```json
{
  "access_token": "ya29.a0...",           // Short-lived access token
  "client_id": "xxx.apps.googleusercontent.com",
  "client_secret": "GOCSPX-xxx",
  "refresh_token": "1//03...",            // Long-lived refresh token
  "token_expiry": "2026-01-14T14:24:50Z", // Access token expiration
  "token_uri": "https://oauth2.googleapis.com/token",
  "scopes": ["https://www.googleapis.com/auth/drive"],
  "invalid": false
}
```

## Token Management

### Access Tokens
- **Lifespan**: ~1 hour
- **Auto-refresh**: Yes, using refresh token
- **Manual refresh**: Not required

### Refresh Tokens
- **Lifespan**: ~6 months (if unused)
- **Renewal**: Automatic on each use
- **Revocation**: Only through Google account settings

## Integration in Your Application

### Current Implementation

The authentication is integrated in `Helpers/google.py`:

```python
def get_drive() -> GoogleDrive:
    gauth = GoogleAuth(SETTINGS_YAML)
    gauth.LoadClientConfigFile(CLIENT_SECRETS)
    
    # Load existing credentials
    gauth.LoadCredentialsFile("token.json")
    
    if gauth.credentials is None:
        # First-time authentication
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh expired token
        gauth.Refresh()
    else:
        # Use existing valid token
        gauth.Authorize()
    
    # Save updated credentials
    gauth.SaveCredentialsFile("token.json")
    return GoogleDrive(gauth)
```

### Usage in Chatbots

- **Cloud Chatbot**: Imported but commented out in `cloud_chabot.py` (line 474)
- **Standalone Chatbot**: Available for use in `standalone_chabot.py`

To activate in your chatbot, uncomment:
```python
download_knowledge_files_from_googleDrive(dest_dir="./Knowledge_base")
```

## Troubleshooting

### Common Issues

1. **"File not found" error**:
   - Ensure `client_secrets.json` exists in `config/` folder
   - Check file path in environment variables

2. **"Permission denied" error**:
   - Verify Google Drive folder permissions
   - Check OAuth scope in `settings.yaml`

3. **"Token expired" error**:
   - Delete `token.json` and re-authenticate
   - Check refresh token validity

4. **Browser doesn't open**:
   - Manually navigate to the displayed URL
   - Check firewall settings for port 8765

### Security Considerations

- **Never commit** `client_secrets.json` or `token.json` to version control
- **Add to .gitignore**:
  ```gitignore
  config/client_secrets.json
  Helpers/token.json
  config/credentials.json
  ```

- **Rotate credentials** periodically through Google Cloud Console
- **Monitor usage** in Google Cloud Console for suspicious activity

## File Locations

```
Project Root/
├── settings.yaml              # OAuth configuration
├── Helpers/
│   ├── google.py             # Authentication logic
│   └── token.json           # Generated tokens (auto-created)
├── config/
│   ├── client_secrets.json  # OAuth credentials (manual)
│   └── credentials.json     # Alternative token storage (optional)
└── .env                     # Environment variables
```

## Next Steps

1. Follow the prerequisites to set up Google Cloud Console
2. Configure your project files
3. Run the authentication script
4. Verify `token.json` is generated
5. Test Google Drive integration in your chatbot

For questions or issues, refer to the [Google Drive API documentation](https://developers.google.com/drive/api/guides/about-auth) or check the project's troubleshooting section.