   set bodyFile to "path/to/your/output.txt"
   set theBody to do shell script "cat " & quoted form of bodyFile
   set theRecipient to "john@theabrahams.ca"

   tell application "Mail"
       set theSubject to "News for " & (do shell script "date +%Y-%m-%d")
       set newMessage to make new outgoing message with properties {subject:theSubject, content:theBody, visible:true}
       tell newMessage
           make new to recipient at end of to recipients with properties {address:theRecipient}
           send
       end tell
   end tell
