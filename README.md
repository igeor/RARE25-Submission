# RARE26 Grand Challenge sanity-check submission

These are the steps used to run the local sanity check and create the Docker
image archive for Grand Challenge from Windows using WSL and Docker Desktop.

## 1. Open WSL

From PowerShell or Command Prompt:

```powershell
wsl
```

Confirm that the shell is running inside WSL:

```bash
uname -a
```

## 2. Go to the repository

```bash
cd /mnt/c/Users/ioann/Documents/GitHub/RARE26-Submission
pwd
```

Expected path:

```text
/mnt/c/Users/ioann/Documents/GitHub/RARE26-Submission
```

## 3. Check Docker Desktop from WSL

Make sure Docker Desktop is running on Windows, then in WSL run:

```bash
docker version
```

Optional Docker test:

```bash
docker run --rm hello-world
```

`hello-world` is only a small Docker test image. It is not part of the
submission.

## 4. Fix Docker credentials in WSL if needed

If Docker fails with an error like:

```text
error getting credentials - err: exit status 1
```

check the Docker config:

```bash
cat ~/.docker/config.json
```

If it contains:

```json
{
  "credsStore": "desktop.exe"
}
```

back it up and replace it with a minimal WSL-compatible config:

```bash
cp ~/.docker/config.json ~/.docker/config.json.bak
printf '{ "auths": {} }\n' > ~/.docker/config.json
cat ~/.docker/config.json
```

Then test pulling the base image:

```bash
docker pull pytorch/pytorch:latest
```

## 5. Normalize and enable the scripts

Run this from the repository root in WSL:

```bash
sed -i 's/\r$//' do_build.sh do_test_run.sh do_save.sh
chmod +x do_build.sh do_test_run.sh do_save.sh
```

## 6. Run the local sanity check

```bash
./do_test_run.sh
```

This builds the Docker image and runs the container offline against:

```text
test/input/interface_0
```

The expected output file is:

```text
test/output/interface_0/stacked-neoplastic-lesion-likelihoods.json
```

Verify it:

```bash
ls -lah test/output/interface_0
cat test/output/interface_0/stacked-neoplastic-lesion-likelihoods.json
```

The JSON should contain numeric likelihood values.

## 7. Save the upload archive

Only run this after the local sanity check succeeds:

```bash
./do_save.sh
```

This rebuilds the image, saves it, and creates files like:

```text
sanity-check-algorithm-v1_YYYY-MM-DD_HH-MM-SS.tar.gz
nmodel.tar.gz
```

For the Grand Challenge Docker algorithm upload, use the newest:

```text
sanity-check-algorithm-v1_*.tar.gz
```

List the generated archives with:

```bash
ls -lh *.tar.gz
```
