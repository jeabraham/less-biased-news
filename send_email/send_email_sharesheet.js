ObjC.import("Foundation");
ObjC.import("AppKit");

function run(argv) {
    var recipientEmail = argv[0]; // Email recipient
    var emailSubject = argv[1];  // Email subject
    var bodyFilePath = argv[2];  // Path to the HTML file
    var senderEmail = argv[3] || null; // Optional sender email

    console.log("Recipient Email: " + recipientEmail);
    console.log("Email Subject: " + emailSubject);
    console.log("HTML File Path: " + bodyFilePath);

    // Read HTML content from file
    var fileManager = $.NSFileManager.defaultManager;
    var htmlPath = $(bodyFilePath).stringByStandardizingPath;

    console.log("Resolved path: " + htmlPath);

    if (!fileManager.fileExistsAtPath(htmlPath)) {
        console.log("Error: HTML file not found at path: " + bodyFilePath);
        return;
    }

    var htmlString = $.NSString.stringWithContentsOfFileEncodingError(htmlPath, $.NSUTF8StringEncoding, null);
    if (!htmlString) {
        console.log("Error: Could not read the HTML content.");
        return;
    }

    // Convert HTML content to attributed string
    var htmlData = htmlString.dataUsingEncoding($.NSUTF8StringEncoding);
    var attributedString = $.NSAttributedString.alloc.initWithHTMLDocumentAttributes(htmlData, null);
    if (!attributedString) {
        console.log("Error: Failed to create NSAttributedString from HTML content.");
        return;
    }

    console.log("HTML content loaded successfully.");

    // Access Mail.app
    var mailApp = Application("Mail");
    mailApp.includeStandardAdditions = true;

    try {
        // Make sure Mail.app is open
        mailApp.activate();

        console.log("Mail.app activated, fetching accounts...");
        console.log("Accounts: " + JSON.stringify(mailApp.accounts));

        // Fetch available sender emails (if sender is not specified)
        var availableAccounts = mailApp.accounts;
        var availableSenders = availableAccounts.map(function (account) {
            return account.emailAddresses();
        }).flat();

        console.log("Available sender addresses: " + availableSenders);

        if (!senderEmail) {
            senderEmail = availableSenders.length > 0 ? availableSenders[0] : null;
            if (!senderEmail) {
                console.log("Error: No sender email configured in Mail.app.");
                return;
            }
        }

        if (!availableSenders.includes(senderEmail)) {
            console.log("Error: Sender email not configured in Mail.app: " + senderEmail);
            return;
        }

        // Create new email
        try {
            console.log("Creating email with subject: " + emailSubject);
            var newMessage = mailApp.OutgoingMessage({
                subject: emailSubject,
                content: attributedString.string, // Plain-text fallback
                visible: true
            });

            console.log("Email draft created successfully.");

            console.log("Adding recipient: " + recipientEmail);
            var recipient = mailApp.Recipient({ address: recipientEmail });
            newMessage.toRecipients.push(recipient);

            console.log("Recipient added successfully.");

            console.log("Setting sender: " + senderEmail);
            newMessage.sender = senderEmail;

            console.log("Email ready to be pushed.");
            mailApp.outgoingMessages.push(newMessage);

            console.log("Email created successfully in Mail.app. Review and send it.");
        } catch (err) {
            console.log("Error creating or sending email: " + err);
            return;
        }
    } catch (err) {
        console.log("Error interacting with Mail.app: " + err);
    }
}
