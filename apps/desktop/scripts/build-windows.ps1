# Build and sign the local Windows Intel desktop package. Publication is separate.
[CmdletBinding()]
param(
    [string]$EngineDir = '',
    [string]$HarnessDir = '',
    [string]$FeedSigningKey = $env:STRATA_FEED_SIGNING_KEY,
    [string]$AppSigningKey = $env:TAURI_SIGNING_PRIVATE_KEY_PATH
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).ProviderPath
$desktop = Join-Path $repo 'apps/desktop'
$resources = Join-Path $desktop 'src-tauri/resources'
$appVersion = (Get-Content -LiteralPath (Join-Path $desktop 'package.json') -Raw | ConvertFrom-Json).version
$installerName = "Strata_${appVersion}_x64-setup.exe"
$python = Join-Path $repo '.venv/Scripts/python.exe'
foreach ($key in @($FeedSigningKey, $AppSigningKey)) {
    if ($key) {
        & $python -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from feed_tool import _inside_git; p=Path(sys.argv[2]); assert p.is_file() and not _inside_git(p), "Signing keys must be existing files outside every Git checkout"' (Join-Path $repo 'packaging') $key
        if ($LASTEXITCODE -ne 0) { throw 'Invalid signing key location.' }
    }
}
if (-not $EngineDir) { $EngineDir = Join-Path $repo 'dist/strata-windows-x64-sycl' }
$EngineDir = (Resolve-Path -LiteralPath $EngineDir).ProviderPath
$manifest = Get-Content -LiteralPath (Join-Path $EngineDir 'engine-component.json') -Raw | ConvertFrom-Json
if ($manifest.backend -ne 'sycl' -or $manifest.platform -ne 'windows-x86_64') { throw 'Choose a Windows SYCL engine package.' }
$env:RUSTUP_TOOLCHAIN = 'stable-x86_64-pc-windows-msvc'
$env:CARGO_TARGET_DIR = Join-Path $repo '.git/desktop-build/target'
$env:CARGO_HOME = Join-Path $repo '.git/desktop-build/cargo-home'
$sysroot = (& rustc --print sysroot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'The stable Windows MSVC Rust toolchain is required.' }
$maps = @("$repo=strata", "$sysroot=rust-toolchain", "$env:USERPROFILE=build-home")
foreach ($key in @('CARGO_HOME', 'RUSTUP_HOME', 'VCToolsInstallDir', 'WindowsSdkDir')) {
    $value = [Environment]::GetEnvironmentVariable($key)
    if ($value) { $maps += "$value=toolchain" }
}
$flags = foreach ($map in $maps) { '--remap-path-prefix=' + $map; '--remap-path-prefix=' + $map.Replace('\', '/') }
$flags += @('-C', 'link-arg=/DEBUG:NONE')
$env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]31
$env:PYTHONUTF8 = '1'
& (Join-Path $PSScriptRoot 'fetch-node.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Node staging failed.' }
& $python (Join-Path $PSScriptRoot 'stage-control.py')
if ($LASTEXITCODE -ne 0) { throw 'Control staging failed.' }
if ($HarnessDir) {
    & node (Join-Path $PSScriptRoot 'stage-harness.mjs') --src $HarnessDir
    if ($LASTEXITCODE -ne 0) { throw 'Harness staging failed.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $resources 'harness/BUNDLE.json'))) {
    throw 'Stage the pinned upstream Harness with stage-harness.mjs before building.'
}
$patchHarness = @'
import { pathToFileURL } from 'node:url';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
const { patchStagedHarnessBudget, HARNESS_BUDGET_PATCH } = await import(pathToFileURL(process.argv[2]).href);
const root = process.argv[3], file = join(root, 'BUNDLE.json');
const bundle = JSON.parse(readFileSync(file, 'utf8'));
const previous = bundle.patches?.find(p => p.id === HARNESS_BUDGET_PATCH.id);
const current = createHash('sha256').update(readFileSync(join(root, HARNESS_BUDGET_PATCH.file))).digest('hex');
const patch = previous?.afterSha256 === current ? previous : patchStagedHarnessBudget(root);
bundle.patches = [...(bundle.patches ?? []).filter(p => p.id !== patch.id), patch];
writeFileSync(file, JSON.stringify(bundle, null, 2));
'@
$patchHarness | & node --input-type=module - (Join-Path $PSScriptRoot 'harness-budget.mjs') (Join-Path $resources 'harness')
if ($LASTEXITCODE -ne 0) { throw 'Harness budget staging failed.' }
& $python -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from strata_control import engine_component as e; e.verify(e.read_manifest(Path(sys.argv[2])), full=True)' $repo $EngineDir
if ($LASTEXITCODE -ne 0) { throw 'Engine package verification failed.' }
$engineBundle = Join-Path $resources 'engine-bundle'
New-Item -ItemType Directory -Force -Path $engineBundle | Out-Null
Copy-Item -LiteralPath $EngineDir -Destination $engineBundle -Recurse -Force
$stagedEngine = Join-Path $engineBundle (Split-Path -Leaf $EngineDir)
if ($FeedSigningKey) {
    & $python (Join-Path $repo 'packaging/feed_tool.py') sign-package $stagedEngine --key $FeedSigningKey
    if ($LASTEXITCODE -ne 0) { throw 'Engine signing failed.' }
}
& $python -c 'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from strata_control import engine_component as e; e.verify_signature(e.read_manifest(Path(sys.argv[2])), {e.ENV_REQUIRE_SIGNED: "1"})' $repo $stagedEngine
if ($LASTEXITCODE -ne 0) { throw 'The bundled engine must have a trusted signature.' }
if ($AppSigningKey) { $env:TAURI_SIGNING_PRIVATE_KEY = $AppSigningKey }
& npm.cmd --prefix (Join-Path $repo 'apps/host') ci
if ($LASTEXITCODE -ne 0) { throw 'Host dependencies failed.' }
& npm.cmd --prefix $desktop ci
if ($LASTEXITCODE -ne 0) { throw 'Desktop dependencies failed.' }
Push-Location $desktop
try {
    & npm.cmd run build:ui
    if ($LASTEXITCODE -ne 0) { throw 'UI build failed.' }
    & npx.cmd tauri build --bundles nsis
    if ($LASTEXITCODE -ne 0) { throw 'Windows desktop build failed.' }
    if ($AppSigningKey) {
        Remove-Item Env:\TAURI_SIGNING_PRIVATE_KEY -ErrorAction SilentlyContinue
        foreach ($installer in Get-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR "release/bundle/nsis/$installerName")) {
            & npx.cmd tauri signer sign --private-key-path $AppSigningKey --app-version $appVersion $installer.FullName
            if ($LASTEXITCODE -ne 0) { throw 'Desktop update signing failed.' }
        }
    }
} finally { Pop-Location }
$audit = @'
import sys, struct, json
from pathlib import Path
binary, resources, *paths = sys.argv[1:]
needles = [variant.lower().encode(enc) for path in paths if path
           for variant in (path, path.replace('\\', '/'), path.replace('\\', '\\\\')) for enc in ('utf-8', 'utf-16le')]
files = [Path(binary), *(p for p in Path(resources).rglob('*') if p.is_file())]
if any(p.suffix.lower() == '.pdb' for p in files):
    raise SystemExit('A debug-symbol PDB file remains in the desktop resources')
exe = Path(binary).read_bytes()
pe = struct.unpack_from('<I', exe, 60)[0]
if struct.unpack_from('<II', exe, pe + 12) != (0, 0) or b'RSDS' in exe:
    raise SystemExit('Desktop debug symbols remain in the release executable')
for file in files:
    data = file.read_bytes().lower()
    if any(needle in data for needle in needles):
        raise SystemExit(f'Build-machine path remains in {file.name}')
sys.path.insert(0, paths[0])
from strata_control.signing import minisign_verify
public = json.loads((Path(resources).parent / 'tauri.conf.json').read_text(encoding='utf-8'))['plugins']['updater']['pubkey']
version = json.loads((Path(resources).parent.parent / 'package.json').read_text(encoding='utf-8'))['version']
for installer in [Path(binary).parent / 'bundle/nsis' / f'Strata_{version}_x64-setup.exe']:
    signature = Path(str(installer) + '.sig').read_text(encoding='utf-8').strip()
    with installer.open('rb') as stream:
        minisign_verify(public, signature, iter(lambda: stream.read(1 << 20), b''))
print('Desktop signatures, executable symbols and bundled resource paths: audit passed.')
'@
$audit | & $python - (Join-Path $env:CARGO_TARGET_DIR 'release/strata-desktop.exe') $resources $repo $env:USERPROFILE $sysroot
if ($LASTEXITCODE -ne 0) { throw 'Build-path audit failed; the installer was not copied to dist.' }
$dist = Join-Path $repo 'dist/desktop-windows-sycl'
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR "release/bundle/nsis/$installerName") -Destination $dist -Force
Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR "release/bundle/nsis/$installerName.sig") -Destination $dist -Force
Write-Output "Local desktop installer: $dist"
