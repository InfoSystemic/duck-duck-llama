#!/usr/bin/env python3
# Read-only routing regression: only mocks health; never connects or runs models.
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
BEFORE = ROOT / 'smeagol-model-route.before.sh'
PROPOSED = ROOT / 'smeagol-model-route.proposed.sh'
FIELDS = ['port', 'status', 'model', 'context', 'kind', 'url']


def route(path, model, live_ports):
    # Override health probing after sourcing declarations. No network calls occur.
    shell = r"""
    set -euo pipefail
    source "$1"
    smeagol_first_live() {
      local port
      for port in "$@"; do
        [[ " $TEST_LIVE_PORTS " == *" $port "* ]] && { printf '%s' "$port"; return 0; }
      done
      return 1
    }
    TEST_LIVE_PORTS="$3"
    SMEAGOL_ROUTE_QUIET=1
    smeagol_route_model "$2"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$SMEAGOL_ROUTE_PORT" "$SMEAGOL_ROUTE_STATUS" "$SMEAGOL_ROUTE_MODEL" "$SMEAGOL_ROUTE_CTX" "$SMEAGOL_ROUTE_KIND" "$SMEAGOL_ROUTE_URL"
    """
    out = subprocess.run(['bash', '-c', shell, 'route-test', str(path), model, live_ports], check=True, text=True, capture_output=True)
    return dict(zip(FIELDS, out.stdout.strip('\n').split('\t')))


def main():
    for path in [BEFORE, PROPOSED]:
        subprocess.run(['bash', '-n', str(path)], check=True)
    aliases = ['glm-5.3-flash', 'glm-flash', 'glm-flash-q4', 'glm-flash-goal', 'GLM-5.3-Flash', 'smeagol/glm-5.3-flash']
    scenarios = [('both_live', '18094 18131', 'live'), ('production_only', '18131', 'live'), ('experimental_only', '18094', 'down'), ('neither_live', '', 'down')]
    cases = []
    for alias in aliases:
        for label, ports, status in scenarios:
            result = route(PROPOSED, alias, ports)
            assert result == dict(port='18131', status=status, model='GLM-5.3-Flash', context='1048576', kind='flash', url='http://127.0.0.1:18131'), result
            cases.append({'alias': alias, 'scenario': label, 'result': result})
    regression_before = route(BEFORE, aliases[0], '18094 18131')
    assert regression_before['port'] == '18094', regression_before
    nonflash = ['glm-sr950', 'glm-sr950-agentic', 'glm-5.3-full', 'qwen3.8-flash-next', 'qwen3.8-27b-q8', 'deepseek-v4.1-flash', 'deepseek-v4-flash', 'deepseek-v4-flash-128k', 'deepseek-v4-flash-fast', 'kimi', 'heretic-pure', 'unknown-model']
    for model in nonflash:
        for ports in ['', '18091', '18091 18092 18083 18081 18172 18170']:
            assert route(BEFORE, model, ports) == route(PROPOSED, model, ports), model
    before = BEFORE.read_text()
    proposed = PROPOSED.read_text()
    start = before.index('    glm-5.3-flash|')
    end = before.index('    glm-sr950-agentic|', start)
    start_new = proposed.index('    glm-5.3-flash|')
    end_new = proposed.index('    glm-sr950-agentic|', start_new)
    assert before[:start] == proposed[:start_new]
    assert before[end:] == proposed[end_new:]
    report = {'status': 'PASS', 'inference_requests': 0, 'network_requests': 0, 'shell_syntax': 'PASS (before and proposed)', 'flash_scenarios': len(cases), 'unchanged_nonflash_scenarios': len(nonflash) * 3, 'baseline_regression_reproduced': regression_before, 'cases': cases}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
