# WhatsApp Integration Setup Guide

This guide will help you set up WhatsApp integration for your Financial Assistant bot.

## Prerequisites

1. A Meta Developer Account
2. A WhatsApp Business Account
3. A server with a public URL (for webhook)

## Step 1: Create a WhatsApp Business App

1. Go to [Meta for Developers](https://developers.facebook.com/)
2. Create a new app and select "Business" as the app type
3. Add the "WhatsApp" product to your app

## Step 2: Get Your Credentials

From your WhatsApp Business API setup, you'll need:

- **Access Token** (`WHATSAPP_TOKEN`)
- **Phone Number ID** (`WHATSAPP_PHONE_NUMBER_ID`) 
- **Verify Token** (`WHATSAPP_VERIFY_TOKEN`) - You create this yourself

## Step 3: Environment Variables

Add these to your `.env` file:

```env
WHATSAPP_TOKEN=your_access_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_here
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token_here
```

## Step 4: Set Up Webhook

1. In your WhatsApp Business API settings, set the webhook URL to:
   ```
   https://your-domain.com/whatsapp_response
   ```

2. Set the verify token to match your `WHATSAPP_VERIFY_TOKEN`

3. Subscribe to these webhook fields:
   - `messages`

## Step 5: Run the Applications

### For Development:

1. **Start the Chainlit app** (existing functionality):
   ```bash
   chainlit run app.py
   ```

2. **Start the WhatsApp webhook server**:
   ```bash
   python whatsapp_app.py
   ```

### For Production:

1. **Chainlit app**:
   ```bash
   chainlit run app.py --host 0.0.0.0 --port 8000
   ```

2. **WhatsApp webhook**:
   ```bash
   uvicorn whatsapp_app:app --host 0.0.0.0 --port 8001
   ```

## Step 6: Test Your Integration

1. Send a text message to your WhatsApp Business number
2. Send an image of a receipt
3. Send a voice message describing a purchase

The bot should respond with processed receipt data and save it to your database.

## Supported Message Types

- **Text**: Natural language purchase descriptions
- **Images**: Receipt photos (automatically analyzed)
- **Audio**: Voice messages (transcribed and processed)

## Example Usage

**Text Message:**
```
"I bought 3 apples for $2 each and 2 bananas for $1.50 total"
```

**Image Message:**
Send a photo of a receipt with optional caption

**Voice Message:**
Record yourself saying: "I spent $25 on groceries today, bought milk, bread, and eggs"


## Security Notes

- Keep your access tokens secure
- Use HTTPS for your webhook URL
- Consider rate limiting for production use