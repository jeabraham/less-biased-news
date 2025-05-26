use framework "Foundation"
use framework "AppKit"
use scripting additions

on run {recipientEmail, emailSubject, bodyFilePath, senderEmail}
    -- Log input values for debugging
    log "Recipient Email: " & recipientEmail
    log "Email Subject: " & emailSubject
    log "HTML File Path: " & bodyFilePath
    log "Sender Email: " & senderEmail

    -- Convert Unix-style path to AppleScript path
    set filePathAsAlias to POSIX file bodyFilePath as alias
    set appleScriptFilePath to filePathAsAlias as text
    log "Converted AppleScript Path: " & appleScriptFilePath

    -- Create an attributed string from the HTML file
    set filePathAsAlias to POSIX file bodyFilePath as alias
    set thisHTML to read filePathAsAlias as «class utf8»
    set theSource to NSString's stringWithString:thisHTML
    set theData to theSource's dataUsingEncoding:NSUTF8StringEncoding
    set anAttributedString to NSAttributedString's alloc()'s initWithHTML:theData documentAttributes:{}

    set aSharingService to NSSharingService's sharingServiceNamed:(NSSharingServiceNameComposeEmail)
    if aSharingService's canPerformWithItems:{"someone@somewhere.com"} then
    set aSharingService's subject to emailSubject
    set aSharingService's recipients to {recipientEmail}
    tell aSharingService to performSelectorOnMainThread:"performWithItems:" withObject:{anAttributedString} waitUntilDone:false
    end if

end run
