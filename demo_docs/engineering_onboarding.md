# Nimbus Robotics — Engineering Onboarding Guide

## Your First Week

Welcome to the Nimbus engineering team! Your first week is about getting set up
and shipping a small change to production. Don't worry about being productive
right away — we expect your first pull request to be tiny, like a typo fix.

### Day 1: Accounts and Access

Your manager will request accounts for GitHub, Slack, Linear, and the internal
VPN. Access is granted through the Okta single sign-on portal. If you are missing
access to a tool, file a request in the #it-help Slack channel. Two-factor
authentication is mandatory on every account.

### Day 2: Local Development Setup

Clone the main monorepo and run `make bootstrap`, which installs dependencies,
sets up pre-commit hooks, and seeds a local database. Our stack is Python for
backend services and TypeScript with React for the web frontend. All services run
locally through Docker Compose.

### Day 3–4: Your First Pull Request

Pick a "good first issue" from Linear. Open a pull request, request review from
your onboarding buddy, and make sure continuous integration passes. Every change
requires at least one approving review before it can be merged. Once merged, our
deployment pipeline ships it to staging automatically.

## Code Review Culture

We review code kindly and thoroughly. Reviewers should respond to a pull request
within one business day. Prefer small pull requests under 400 lines of change;
large ones are hard to review and slow to merge. Use "nit:" to mark non-blocking
suggestions so authors know what is optional.

## Deployment

Nimbus ships to production multiple times per day. Merges to the main branch are
deployed to staging automatically, and promoted to production after a smoke test
passes. Any engineer can trigger a production deploy, but production deploys are
frozen on Fridays after 14:00 to avoid weekend incidents.
