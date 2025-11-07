# Quick Wins

1. **Fix cache lookups with retrieval parameters**  
   ```python
   # clockify_support_cli_final.py
-    def get(self, question: str):
+    def get(self, question: str, params: dict | None = None):
         """Retrieve cached answer if available and not expired."""
-        key = self._hash_question(question)
+        key = self._hash_question(question, params)
   ```

2. **Allow cache writes with retrieval parameters**  
   ```python
   # clockify_support_cli_final.py
-    def put(self, question: str, answer: str, metadata: dict):
+    def put(self, question: str, answer: str, metadata: dict, params: dict | None = None):
         """Store answer in cache."""
-        key = self._hash_question(question)
+        key = self._hash_question(question, params)
   ```

3. **Remove duplicate ensure_index_ready definition**  
   ```python
   # clockify_support_cli_final.py
-@deprecated
def ensure_index_ready(retries=0):
-    artifacts = [FILES["chunks"], FILES["emb"], FILES["meta"], FILES["bm25"], FILES["index_meta"]]
-    artifacts_ok = all(os.path.exists(fname) for fname in artifacts)
+def ensure_index_ready(retries=0):
+    # single authoritative implementation (keep rest of existing body)
   ```

4. **Seed FAISS training on arm64**  
   ```python
   # clockify_support_cli_final.py
-            if len(vecs) >= m1_train_size:
-                train_indices = np.random.choice(len(vecs), m1_train_size, replace=False)
+            rng = np.random.default_rng(DEFAULT_SEED)
+            if len(vecs) >= m1_train_size:
+                train_indices = rng.choice(len(vecs), m1_train_size, replace=False)
   ```

5. **Seed FAISS training on other platforms**  
   ```python
   # clockify_support_cli_final.py
-        train_indices = np.random.choice(len(vecs), train_size, replace=False)
+        rng = np.random.default_rng(DEFAULT_SEED)
+        train_indices = rng.choice(len(vecs), train_size, replace=False)
   ```

6. **Warn when warm-up fails**  
   ```python
   # clockify_support_cli_final.py
-    except Exception as e:
-        logger.debug(f"Warm-up skipped: {e}")
+    except Exception as e:
+        logger.warning("warmup failed: %s", e)
   ```

7. **Avoid double exit in ensure_index_ready**  
   ```python
   # clockify_support_cli_final.py
-        else:
-            logger.error("knowledge_full.md not found")
-            sys.exit(1)
-            raise SystemExit(1)
+        else:
+            logger.error("knowledge_full.md not found")
+            sys.exit(1)
   ```

8. **Skip chunk bodies in logs when disabled**  
   ```python
   # clockify_support_cli_final.py
-            entry = {
-                "id": chunk_id,
-                "pack_rank": pack_rank,
-                "dense": info["dense"],
-                "bm25": info["bm25"],
-                "hybrid": info["hybrid"],
-                "chunk": chunk,
-            }
+            entry = {
+                "id": chunk_id,
+                "pack_rank": pack_rank,
+                "dense": info["dense"],
+                "bm25": info["bm25"],
+                "hybrid": info["hybrid"],
+            }
+            if LOG_QUERY_INCLUDE_ANSWER:
+                entry["chunk"] = chunk
   ```

9. **Expose retrieval env vars in README**  
   ```markdown
   | Variable | Default | Purpose |
   |----------|---------|---------|
   | `ALPHA` | `0.5` | Hybrid weight between BM25 and dense scores |
   | `ANN_NLIST` | `64` | FAISS IVF cluster count |
   | `CACHE_TTL` | `3600` | Query cache TTL (seconds) |
   ```

10. **Raise warning when reranker JSON parse fails**  
    ```python
    # clockify_support_cli_final.py
-        except json.JSONDecodeError:
-            logger.debug("info: rerank=fallback reason=json")
+        except json.JSONDecodeError:
+            logger.warning("rerank fallback: invalid JSON output")
    ```
