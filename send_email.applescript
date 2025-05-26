on run {recipientEmail, emailSubject, bodyFilePath, senderEmail}
    -- Read the email body from the file
    set theBody to do shell script "cat " & quoted form of bodyFilePath

    tell application "Mail"
        -- Create a new email
        set newMessage to make new outgoing message with properties {subject:emailSubject, visible:true}

        -- Set the recipient
        tell newMessage
            make new to recipient at end of to recipients with properties {address:recipientEmail}
            set content to theBody
        end tell

        end tell

        -- Set the sender email address
        set newMessage's sender to senderEmail

        -- Send the email
        send newMessage
    end tell
end run
