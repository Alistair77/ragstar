# Nimbus Robotics — Security Incident Response Policy

## Severity Levels

Nimbus classifies every security incident into one of four severity levels. The
severity determines who is paged and how quickly the team must respond.

- **SEV-1 (Critical):** Active data breach, customer data exposure, or full
  production outage. Response required within 15 minutes, 24/7.
- **SEV-2 (High):** Partial outage, exploited vulnerability with limited blast
  radius, or failed critical backup. Response required within 1 hour.
- **SEV-3 (Medium):** Non-exploited vulnerability, degraded performance, or an
  isolated internal service failure. Response required within 1 business day.
- **SEV-4 (Low):** Minor misconfiguration or cosmetic issue with no security
  impact. Handled during normal work.

## Escalation Process

When a SEV-1 incident is declared, the on-call engineer must immediately page the
Incident Commander and the Head of Security through the PagerDuty "critical"
policy. A dedicated incident channel is opened in Slack, and a bridge call is
started within 15 minutes. Customer Support is notified so that a status page
update can be published within 30 minutes of declaration.

For SEV-2 incidents, the on-call engineer pages the Incident Commander only. For
SEV-3 and SEV-4, the issue is filed as a ticket and triaged in the next daily
standup.

## Postmortems

Every SEV-1 and SEV-2 incident requires a blameless postmortem within five
business days. The postmortem documents the timeline, root cause, customer
impact, and concrete action items with owners. Postmortems are shared company-wide
to reinforce a culture of learning rather than blame.

## Data Breach Notification

If customer personal data is confirmed exposed, Nimbus must notify affected
customers within 72 hours, in line with GDPR requirements. The Legal team owns
external communication and regulatory reporting.
