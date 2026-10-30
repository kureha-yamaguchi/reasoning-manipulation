'''
Minimal RunPod GraphQL client for one-off GPU pods (used by bash_scripts/runpod/).

The API key is read from $RUNPOD_API_KEY or ~/.config/runpod/api_key and never written out.
Pod metadata (id, price, ssh endpoint) goes to --state, a JSON file.

uv run python -m utils.runpod_pod price  --gpu "NVIDIA H200" --count 8
uv run python -m utils.runpod_pod launch --gpu "NVIDIA H200,NVIDIA H200 NVL" --count 8 --state S --max_wait_h 24
uv run python -m utils.runpod_pod ssh    --state S        # prints "ip port" once SSH is reachable
uv run python -m utils.runpod_pod status --state S
uv run python -m utils.runpod_pod stop   --state S        # keeps the volume
'''

import argparse
import json
import os
import time
import urllib.error
import urllib.request

API = 'https://api.runpod.io/graphql'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'


def api_key() -> str:
    key = os.environ.get('RUNPOD_API_KEY')
    if not key:
        with open(os.path.expanduser('~/.config/runpod/api_key')) as f:
            key = f.read().strip()
    return key


def gql(query: str, variables: dict | None = None) -> dict:
    req = urllib.request.Request(API, data=json.dumps({'query': query, 'variables': variables or {}}).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key()}',
                                          'User-Agent': 'reasoning-manipulation/0.1'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:500]}') from None
    if out.get('errors'):
        raise RuntimeError(out['errors'])
    return out['data']


def price(gpu: str, count: int) -> dict | None:
    q = '''query($ids: [String!], $n: Int!) { gpuTypes(input: {ids: $ids}) {
             lowestPrice(input: {gpuCount: $n, secureCloud: true}) { uninterruptablePrice stockStatus } } }'''
    lp = gql(q, {'ids': [gpu], 'n': count})['gpuTypes'][0]['lowestPrice']
    return lp if lp and lp.get('uninterruptablePrice') else None


def deploy(gpu: str, count: int, name: str, volume_gb: int, public_key: str) -> dict:
    q = '''mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) {
             id costPerHr machineId desiredStatus } }'''
    inp = {'cloudType': 'SECURE', 'gpuCount': count, 'gpuTypeId': gpu, 'name': name, 'imageName': IMAGE,
           'containerDiskInGb': 100, 'volumeInGb': volume_gb, 'volumeMountPath': '/workspace', 'ports': '22/tcp',
           'startSsh': True, 'supportPublicIp': True, 'env': [{'key': 'PUBLIC_KEY', 'value': public_key}]}
    return gql(q, {'input': inp})['podFindAndDeployOnDemand']


def pod(pod_id: str) -> dict | None:
    q = '''query($id: String!) { pod(input: {podId: $id}) { id desiredStatus costPerHr
             runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } }'''
    return gql(q, {'id': pod_id})['pod']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cmd', choices=['price', 'launch', 'ssh', 'status', 'stop'])
    parser.add_argument('--gpu', default='NVIDIA H200', help='Comma list for launch: first type in stock wins')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--state', default=None)
    parser.add_argument('--name', default='r1-quadrants')
    parser.add_argument('--volume_gb', type=int, default=1400)
    parser.add_argument('--max_wait_h', type=float, default=24.0)
    parser.add_argument('--poll_s', type=int, default=600)
    args = parser.parse_args()

    if args.cmd == 'price':
        print(json.dumps(price(args.gpu, args.count)))
        return
    if args.cmd == 'launch':
        with open(os.path.expanduser('~/.ssh/id_ed25519.pub')) as f:
            public_key = f.read().strip()
        deadline = time.time() + args.max_wait_h * 3600
        polls = 0
        while True:
            polls += 1
            try:
                gpu, lp = next(((g, p) for g in args.gpu.split(',') if (p := price(g, args.count))), (None, None))
            except (RuntimeError, OSError) as e:  # transient API/network errors must not end a long wait
                print(f'{time.strftime("%F %T")} poll failed: {e}', flush=True)
                time.sleep(max(args.poll_s, 30))
                continue
            if lp:
                print(f'{time.strftime("%F %T")} {args.count}x {gpu} in stock ({lp}); deploying', flush=True)
                try:
                    created = deploy(gpu, args.count, args.name, args.volume_gb, public_key)
                    state = {'pod_id': created['id'], 'cost_per_hr': created['costPerHr'], 'gpu': gpu,
                             'count': args.count, 'created': time.time(), 'image': IMAGE}
                    with open(args.state, 'w') as f:
                        json.dump(state, f, indent=2)
                    print(json.dumps(state), flush=True)
                    return
                except RuntimeError as e:  # stock vanished between query and deploy
                    print(f'{time.strftime("%F %T")} deploy failed: {e}', flush=True)
            elif polls % max(1, 60 // args.poll_s) == 1:  # log ~once a minute, with 1x stock (fragmentation)
                try:
                    singles = {g: (price(g, 1) or {}).get('stockStatus') for g in args.gpu.split(',')}
                except (RuntimeError, OSError):
                    singles = 'unavailable'
                print(f'{time.strftime("%F %T")} no {args.count}x {args.gpu} in stock; 1x stock {singles}', flush=True)
            if time.time() > deadline:
                raise SystemExit(f'No {args.count}x {args.gpu} within {args.max_wait_h} h')
            time.sleep(args.poll_s)

    with open(args.state) as f:
        state = json.load(f)
    if args.cmd == 'status':
        print(json.dumps(pod(state['pod_id'])))
    elif args.cmd == 'stop':
        print(json.dumps(gql('mutation($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }',
                             {'id': state['pod_id']})))
    elif args.cmd == 'ssh':
        for _ in range(120):  # up to 20 min for image pull and sshd
            info = pod(state['pod_id'])
            ports = ((info or {}).get('runtime') or {}).get('ports') or []
            ssh = [p for p in ports if p['privatePort'] == 22 and p['isIpPublic']]
            if ssh:
                print(ssh[0]['ip'], ssh[0]['publicPort'])
                return
            time.sleep(10)
        raise SystemExit('SSH endpoint not available after 20 min')


if __name__ == '__main__':
    main()
