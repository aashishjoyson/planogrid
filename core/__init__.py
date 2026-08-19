"""Contracts and pure logic shared by every other package in this repo.

core/ depends on nothing else first-party. models/, training/, inference/,
and app/ depend on core/ — never the other way around. See DESIGN.md §4.1.
"""
