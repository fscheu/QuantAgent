AGENTE CODEX: ENTREGA SIN GIT (manda sobre el resto del prompt donde se contradigan)
Tu aislamiento deja `.git` en solo lectura. No corras `git checkout -b`, `git commit`, `git push`, `gh pr create`
ni `gh pr comment`: fallan. Lo hace el wrapper cuando termina tu turno. Donde el prompt pida esos pasos, dejá
en la raíz del worktree la carpeta `.loop-delivery/` con:
- `ticket`: el ID completo (`QuantAgent-xxxx`). Solo en modo nuevo.
- `commit-msg`: el mensaje `<tipo>(<ID>): <qué>`.
- `pr-body.md`: el cuerpo completo de `.github/pull_request_template.md`. En modo cambio, el comentario que
  explica qué cambiaste.
- `labels` (opcional): `bloqueo` o `particion`, uno por línea.
El wrapper commitea todo el árbol de trabajo: no dejes archivos temporales. El tamaño se mide con
`git diff origin/<base>` sobre el árbol de trabajo, sin commit. `git status`, `git diff` y `bd` funcionan.
No esperes el CI: en el cuerpo anotá "CI: lo corre GitHub al abrir el PR". El commit local previo a la
verificación independiente no aplica; si no podés lanzar el verificador, anotá "Verificador independiente: no corrió".
