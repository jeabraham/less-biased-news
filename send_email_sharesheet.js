ObjC.import("Foundation");
ObjC.import("AppKit");

// Function to run the script
function run(argv) {
    var recipientEmail = argv[0];
    var emailSubject = argv[1];
    var bodyFilePath = argv[2];
    var senderEmail = argv[3];

    console.log("Recipient Email: " + recipientEmail);
    console.log("Email Subject: " + emailSubject);
    console.log("HTML File Path: " + bodyFilePath);
    console.log("Sender Email: " + senderEmail);

    // Read the contents of the HTML file
    var fileManager = $.NSFileManager.defaultManager;
    var htmlPath = $(bodyFilePath).stringByStandardizingPath;
    if (!fileManager.fileExistsAtPath(htmlPath)) {
        console.log("Error: HTML file does not exist at path: " + bodyFilePath);
        return;
    }

    var htmlContent = $.NSString.stringWithContentsOfFileEncodingError(htmlPath, $.NSUTF8StringEncoding, null);
    if (!htmlContent) {
        console.log("Error: Could not read the HTML file.");
        return;
    }

    // Convert HTML content to NSAttributedString for rich formatting
    var data = htmlContent.dataUsingEncoding($.NSUTF8StringEncoding);
    var attributedString = $.NSAttributedString.alloc.initWithHTMLDocumentAttributes(data, null);

    // Initialize NSSharingService for composing email
    var sharingService = $.NSSharingService.sharingServiceNamed($.NSSharingServiceNameComposeEmail);
    if (sharingService.canPerformWithItems([recipientEmail])) {
        sharingService.setSubject(emailSubject);
        sharingService.setRecipients([recipientEmail]);

        // Send the email with the attributed HTML content
        sharingService.performWithItems([attributedString]);
        console.log("Email created and ready to send.");
    } else {
        console.log("Error: Unable to send email via sharing service.");
    }
}
