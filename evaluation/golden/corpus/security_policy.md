# Lumen Analytics Information Security Policy

## Access control
- Passwords must be at least 14 characters long.
- Passwords must be rotated every 180 days for privileged accounts only.
- Multi-factor authentication is mandatory for all employees and uses hardware security keys.
- Access rights are reviewed quarterly by the system owner.
- Production database access requires approval from two people.
- Shared accounts are prohibited.
- Inactive accounts are disabled after 45 days without login.

## Data handling
- Data is classified into four levels: Public, Internal, Confidential, and Restricted.
- Customer data is classified as Confidential by default.
- Restricted data includes encryption keys and payment card numbers.
- All data at rest is encrypted with AES-256.
- All data in transit uses TLS 1.2 or higher.
- Restricted data must never be stored on laptops or sent by email.
- Customer data may only be processed in data centers located in the European Union.

## Incident response
- Security incidents must be reported to the security team within 1 hour of discovery.
- Critical incidents are acknowledged within 15 minutes by the on-call security engineer.
- Affected customers are notified within 72 hours of a confirmed personal data breach.
- A post-incident review must be completed within 5 business days.
- The on-call rotation for security engineers changes every Monday at 09:00.

## Backups and continuity
- Databases are backed up every 6 hours and backups are retained for 35 days.
- Backups are stored in a different region from the primary data center.
- The recovery time objective is 4 hours and the recovery point objective is 6 hours.
- Disaster recovery tests are conducted twice a year.

## Vendors and vulnerability management
- New vendors that process customer data require a security review before contracts are signed.
- Critical vulnerabilities must be patched within 7 days.
- High severity vulnerabilities must be patched within 30 days.
- Penetration tests by an external firm are performed annually.
- The bug bounty program pays between 100 and 5,000 euros depending on severity.
