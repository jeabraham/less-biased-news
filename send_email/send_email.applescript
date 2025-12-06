on run {recipientEmail, emailSubject, bodyFilePath, senderEmail, shouldAttach}
    -- Log input values for debugging
    log "Recipient Email: " & recipientEmail
    log "Email Subject: " & emailSubject
    log "HTML File Path: " & bodyFilePath
    log "Sender Email: " & senderEmail
    log "Should Attach File: " & shouldAttach

    -- Convert Unix-style path to AppleScript path
    set filePathAsAlias to POSIX file bodyFilePath as alias
    set appleScriptFilePath to filePathAsAlias as text
    log "Converted AppleScript Path: " & appleScriptFilePath

    set theSignatureText to "
--
Processed by Less Biased News
https://github.com/jeabraham/less-biased-news/
"

    set theDelay to 1

    -- Load the HTML body content from the file
    set htmlContent to ""
    try
        set htmlContent to read filePathAsAlias as «class utf8»
        log "HTML content successfully loaded."
    on error errMsg
        log "Error reading HTML content: " & errMsg
    end try

    -- Create email in Mail.app
    tell application "Mail"
        -- Create a new outgoing message and set the HTML content
        log "Creating new outgoing message..."
        set newMessage to make new outgoing message with properties ¬
            {subject:emailSubject, visible:true, content:htmlContent}

        -- Add recipient to the email
        tell newMessage
            log "Adding recipient..."
            make new to recipient at end of to recipients with properties {address:recipientEmail}
        end tell

        -- Check shouldAttach and add attachment if true
        if shouldAttach is "true" or shouldAttach is "True" then
            tell newMessage
                log "Attaching file..."
                try
                    set theAttachment to alias "Macintosh HD:Users:jabraham:Development:less-biased-news:news_test_email.html"
                    make new attachment with properties {file name:theAttachment} at after last paragraph
                    delay theDelay
                    log "Attachment added successfully."
                on error errMsg
                    log "Error attaching file: " & errMsg
                end try
                log "Attachment logic complete."
            end tell
        else
            log "Attachment not included due to shouldAttach flag."
        end if

        -- Add signature text
        tell newMessage to make new paragraph at end of paragraphs of content with data theSignatureText

        -- Set the sender email address
        set newMessage's sender to senderEmail

        -- Uncomment this line to send the email automatically
        send newMessage
    end tell
end run
