"""Build a chat-formatted SFT dataset for React Native code generation.

Input  : data/rn_seed.jsonl with one {"instruction": ..., "code": ...} per line.
Output : data/train.jsonl + data/eval.jsonl with a "messages" field ready for
         TRL's SFTTrainer chat template.

If data/rn_seed.jsonl is missing, a tiny synthetic seed is written so the
pipeline runs end-to-end. Replace it with a real corpus before serious training.
"""
import json
import os
import random
from pathlib import Path

from config import CFG

SYSTEM_PROMPT = (
    "You are an expert React Native engineer. Generate idiomatic, production-"
    "quality React Native code using functional components, hooks, and "
    "TypeScript when appropriate. Prefer the core RN API and well-known "
    "libraries (react-navigation, react-native-reanimated). Return only code "
    "in a single fenced block unless explanation is requested."
)

SYNTHETIC_SEED = [
    {
        "instruction": "Create a React Native screen with a FlatList of users fetched from /api/users, with pull-to-refresh.",
        "code": (
            "import React, { useEffect, useState, useCallback } from 'react';\n"
            "import { FlatList, RefreshControl, Text, View, StyleSheet } from 'react-native';\n\n"
            "export default function UsersScreen() {\n"
            "  const [users, setUsers] = useState([]);\n"
            "  const [refreshing, setRefreshing] = useState(false);\n\n"
            "  const load = useCallback(async () => {\n"
            "    setRefreshing(true);\n"
            "    const res = await fetch('/api/users');\n"
            "    setUsers(await res.json());\n"
            "    setRefreshing(false);\n"
            "  }, []);\n\n"
            "  useEffect(() => { load(); }, [load]);\n\n"
            "  return (\n"
            "    <FlatList\n"
            "      data={users}\n"
            "      keyExtractor={(u) => String(u.id)}\n"
            "      renderItem={({ item }) => (\n"
            "        <View style={styles.row}><Text>{item.name}</Text></View>\n"
            "      )}\n"
            "      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}\n"
            "    />\n"
            "  );\n"
            "}\n\n"
            "const styles = StyleSheet.create({ row: { padding: 16 } });\n"
        ),
    },
    {
        "instruction": "Write a custom React Native hook useDebouncedValue<T>(value, delay).",
        "code": (
            "import { useEffect, useState } from 'react';\n\n"
            "export function useDebouncedValue<T>(value: T, delay = 300): T {\n"
            "  const [debounced, setDebounced] = useState(value);\n"
            "  useEffect(() => {\n"
            "    const id = setTimeout(() => setDebounced(value), delay);\n"
            "    return () => clearTimeout(id);\n"
            "  }, [value, delay]);\n"
            "  return debounced;\n"
            "}\n"
        ),
    },
    {
        "instruction": "Build a React Native login form with email + password validation and a submit button that disables while loading.",
        "code": (
            "import React, { useState } from 'react';\n"
            "import { View, TextInput, Button, Text, StyleSheet } from 'react-native';\n\n"
            "export default function LoginForm({ onSubmit }: { onSubmit: (e: string, p: string) => Promise<void> }) {\n"
            "  const [email, setEmail] = useState('');\n"
            "  const [password, setPassword] = useState('');\n"
            "  const [loading, setLoading] = useState(false);\n"
            "  const [error, setError] = useState<string | null>(null);\n\n"
            "  const valid = /.+@.+\\..+/.test(email) && password.length >= 8;\n\n"
            "  const submit = async () => {\n"
            "    setLoading(true); setError(null);\n"
            "    try { await onSubmit(email, password); }\n"
            "    catch (e: any) { setError(e.message ?? 'Login failed'); }\n"
            "    finally { setLoading(false); }\n"
            "  };\n\n"
            "  return (\n"
            "    <View style={styles.c}>\n"
            "      <TextInput style={styles.i} placeholder='Email' autoCapitalize='none' value={email} onChangeText={setEmail} />\n"
            "      <TextInput style={styles.i} placeholder='Password' secureTextEntry value={password} onChangeText={setPassword} />\n"
            "      {error && <Text style={styles.err}>{error}</Text>}\n"
            "      <Button title={loading ? 'Signing in...' : 'Sign in'} onPress={submit} disabled={!valid || loading} />\n"
            "    </View>\n"
            "  );\n"
            "}\n\n"
            "const styles = StyleSheet.create({\n"
            "  c: { padding: 16, gap: 12 },\n"
            "  i: { borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 12 },\n"
            "  err: { color: 'red' },\n"
            "});\n"
        ),
    },
]


def write_synthetic_seed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in SYNTHETIC_SEED:
            f.write(json.dumps(ex) + "\n")


def to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": f"```tsx\n{example['code'].rstrip()}\n```"},
        ]
    }


def main() -> None:
    seed_path = Path(CFG.seed_dataset_path)
    if not seed_path.exists():
        print(f"[prepare_data] {seed_path} missing — writing synthetic seed.")
        write_synthetic_seed(seed_path)

    rows = [json.loads(line) for line in seed_path.open()]
    random.Random(0).shuffle(rows)
    n_eval = max(1, int(len(rows) * CFG.eval_ratio))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    os.makedirs(CFG.data_dir, exist_ok=True)
    for path, items in [(CFG.train_path, train_rows), (CFG.eval_path, eval_rows)]:
        with open(path, "w") as f:
            for ex in items:
                f.write(json.dumps(to_messages(ex)) + "\n")
        print(f"[prepare_data] wrote {len(items)} rows -> {path}")


if __name__ == "__main__":
    main()
