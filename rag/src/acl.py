"""
Access Control Layer (ACL)

Every chunk carries a set of permission tokens.  At retrieval time the
store pre-filters to only chunks the requesting user is allowed to see.

Permission token format
-----------------------
  "user:<user_id>"      — only this specific user
  "team:<team_name>"    — anyone in this team
  "role:<role_name>"    — anyone with this role
  "team:all"            — everyone (public document)

Example
-------
  chunk.metadata["permissions"] = ["team:finance", "team:product"]
  user in team:finance  → can see this chunk
  user in team:engineering → cannot

Why pre-filter, not post-filter
--------------------------------
If you retrieve top-100 then remove forbidden results you may return
0 results even when 50 permitted results exist just outside the top-100.
Pre-filtering ranks only within the permitted set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UserContext:
    """Represents the identity of the user making a query."""
    user_id: str
    teams: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    @property
    def permission_tokens(self) -> set:
        """All tokens this user satisfies."""
        tokens = {f"user:{self.user_id}", "team:all"}
        tokens.update(f"team:{t}" for t in self.teams)
        tokens.update(f"role:{r}" for r in self.roles)
        return tokens


def can_access(user: UserContext, permissions: List[str]) -> bool:
    """
    Return True if *user* satisfies at least one permission token.

    An empty permissions list means the document is public.
    A superuser (user_id == "__superuser__") bypasses all ACL checks.
    """
    if user.user_id == "__superuser__":
        return True
    if not permissions:
        return True
    return bool(user.permission_tokens & set(permissions))


# Convenience: a superuser that can see everything
SUPERUSER = UserContext(user_id="__superuser__", teams=["all"], roles=["admin"])
