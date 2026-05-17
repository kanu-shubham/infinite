"""
acl.py — Access Control List.

Every user has a set of permission tokens.
Every chunk has a set of required permission tokens.
A user can read a chunk only if their tokens cover ALL of the chunk's required tokens.
"""
from dataclasses import dataclass, field


SUPERUSER = "__superuser__"   # special user_id that bypasses all checks


@dataclass
class UserContext:
    user_id: str
    permissions: set[str] = field(default_factory=set)
    # Example: {"hr_docs", "engineering_docs", "finance_docs"}


def can_access(user: UserContext, chunk_permissions: set[str]) -> bool:
    """
    Returns True if the user may see this chunk.

    Rules:
      1. Superuser always passes (admin tooling, batch jobs).
      2. If the chunk has no permission requirements → public, everyone passes.
      3. Otherwise the user must hold ALL required tokens.
    """
    if user.user_id == SUPERUSER:
        return True
    if not chunk_permissions:
        return True
    return chunk_permissions.issubset(user.permissions)
