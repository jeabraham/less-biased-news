use framework "Foundation"
use framework "AppKit"
use scripting additions

set recipientEmail to "john@theabrahams.ca"
set emailSubject to "News Today"
set bodyFilePath to "/Users/jabraham/Development/less-biased-news/output/news_2025-05-26_email.html"
set senderEmail to "john@theabrahams.ca"

--on run {recipientEmail, emailSubject, bodyFilePath, senderEmail}

-- Log input values for debugging
log "Recipient Email: " & recipientEmail
log "Email Subject: " & emailSubject
log "HTML File Path: " & bodyFilePath
log "Sender Email: " & senderEmail

-- Ensure the file path is a valid POSIX path
set filePathAsURL to current application's NSURL's fileURLWithPath:bodyFilePath
if filePathAsURL = missing value then
	log "Error: Invalid file path."
	return
end if
log "Resolved POSIX Path: " & bodyFilePath

-- Read the HTML file content
set fileManager to current application's NSFileManager's defaultManager()
set isReadable to fileManager's isReadableFileAtPath:bodyFilePath
if isReadable as boolean is false then
	log "Error: HTML file is not readable."
	return
end if

set thisHTML to current application's NSString's stringWithContentsOfFile:bodyFilePath encoding:(current application's NSUTF8StringEncoding) |error|:(missing value)
if thisHTML = missing value then
	log "Error: Unable to read HTML content."
	return
end if
log "HTML Content Loaded: " & (thisHTML's substringToIndex:100) -- Log only the first 100 characters

-- Create an attributed string from the HTML
set theData to thisHTML's dataUsingEncoding:(current application's NSUTF8StringEncoding)
set {anAttributedString, errorInfo} to current application's NSAttributedString's alloc()'s initWithData:theData options:{NSDocumentTypeDocumentAttribute:(current application's NSHTMLTextDocumentType)} documentAttributes:(missing value) |error|:(reference)
if anAttributedString = missing value then
	log "Error: Failed to create NSAttributedString - " & errorInfo
	return
end if
log "Attributed String Created Successfully"

-- Create and configure the sharing service
set aSharingService to current application's NSSharingService's sharingServiceNamed:(current application's NSSharingServiceNameComposeEmail)
if aSharingService's canPerformWithItems:{recipientEmail} then
	-- Assign subject and recipient
	aSharingService's setSubject:emailSubject
	aSharingService's setRecipients:{recipientEmail}
	
	-- Perform the sharing service
	aSharingService's performSelectorOnMainThread:"performWithItems:" withObject:{anAttributedString} waitUntilDone:true
	log "Email successfully passed to SharingService"
else
	log "Error: Unable to configure sharing service with recipient."
	return
end if

-- end run
