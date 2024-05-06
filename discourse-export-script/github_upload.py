from argparse import ArgumentParser
from asyncio import gather, run, sleep
from datetime import datetime
from json import load, loads
from os import listdir
from os.path import isfile, join
from re import findall, sub
from shutil import move
from typing import Any, Dict

from aiohttp import ClientSession

now = datetime.now
fromtimestamp = datetime.fromtimestamp

# constants
POST_LINK_ID = "f4706281-cc60-4ff0-a0b6-b803683cc24b"


async def main(github_issue_number: int, github_token: str) -> None:
    links = {}
    for idx, filename in enumerate(sorted(listdir("./discourse-export"))):
        if isfile(join("./discourse-export", filename)):
            id, rest = filename.split("_")
            slug, _ = rest.split(".")
            with open(f"./discourse-export/{id}_{slug}.json", "r") as file:
                links[slug] = {
                    "id": id,
                    "idx": idx,
                    "dests": set([slug[:-1] for slug in findall(f'{POST_LINK_ID}/(.*?)"', file.read())]),
                }
    for slug, data in links.items():
        for dest in data["dests"]:
            dest_data = links.get(dest, None)
            if dest_data is None:
                # TODO sometimes the dest here is grabbing too much text after the link, whats up with that
                """
broken link: illegalargumentexception->error-summary-illegalargumentexception-requirement-failed/2474/8</a></p>\n<p>Could you have a look what is going on here ?</p>\n\n\n```python\nIn [13]: mtf = hl.filter_intervals(mt, [hl.parse_locus_interval('chr19:50927163-50927163', reference_genome=rg)])\n\nIn [14]: mtf.show()\n---------------------------------------------------------------------------\nFatalError                                Traceback (most recent call last)\nFile /usr/local/lib/python3.11/dist-packages/IPython/core/formatters.py:711, in PlainTextFormatter.__call__(self, obj)\n    704 stream = StringIO()\n    705 printer = pretty.RepresentationPrinter(stream, self.verbose,\n    706     self.max_width, self.newline,\n    707     max_seq_length=self.max_seq_length,\n    708     singleton_pprinters=self.singleton_printers,\n    709     type_pprinters=self.type_printers,\n    710     deferred_pprinters=self.deferred_printers)\n--> 711 printer.pretty(obj)\n    712 printer.flush()\n    713 return stream.getvalue()\n\nFile /usr/local/lib/python3.11/dist-packages/IPython/lib/pretty.py:411, in RepresentationPrinter.pretty(self, obj)\n    408                         return meth(obj, self, cycle)\n    409                 if cls is not object \\\n    410                         and callable(cls.__dict__.get('__repr__')):\n--> 411                     return _repr_pprint(obj, self, cycle)\n    413     return _default_pprint(obj, self, cycle)\n    414 finally:\n\nFile /usr/local/lib/python3.11/dist-packages/IPython/lib/pretty.py:779, in _repr_pprint(obj, p, cycle)\n    777  (https://github.com/hail-is/hail/issues/3595)
broken link: illegalargumentexception->error-summary-illegalargumentexception-requirement-failed (https://github.com/hail-is/hail/issues/3595)
broken link: ploidy-error-when-trying-to-extract-x-chromosome-snps->error-summary-hailexception-only-support-ploidy-2-and-unphased-found-1-1 (https://github.com/hail-is/hail/issues/3599)
                """
                print(
                    f"broken link: {slug}->{dest} (https://github.com/hail-is/hail/issues/{github_issue_number + data['idx']})"
                )
            else:
                with open(f"./discourse-export/{data['id']}_{slug}.json", "r") as file:
                    json = sub(
                        f'{POST_LINK_ID}/{dest}"',
                        f"https://github.com/hail-is/hail/issues/{github_issue_number + dest_data['idx']}\\\"",
                        file.read(),
                    )
                with open(f"./discourse-export/{data['id']}_{slug}.json", "w") as file:
                    file.write(json)
    async with ClientSession() as session:
        count = 0
        for issue in sorted([{"slug": slug, **data} for slug, data in links.items()], key=lambda x: x["idx"]):
            with open(f"./discourse-export/{issue['id']}_{issue['slug']}.json", "r") as file:
                topic = load(file)
            uploaded = False
            while not uploaded:
                uploaded = next(iter(await gather(create_issue(topic, session, github_token))))
            move(
                f"./discourse-export/{issue['id']}_{issue['slug']}.json",
                f"./uploaded/{issue['id']}_{issue['slug']}.json",
            )
            if count < 19:
                count += 1
            else:
                count = 0
                print("Waiting for 2 minutes...")
                await sleep(120)


async def create_issue(topic: Dict[str, Any], session: ClientSession, github_token: str) -> bool:
    async with session.post(
        "https://api.github.com/repos/iris-garden/test-process/issues",
        json={"title": topic["title"], "body": topic["html"], "labels": ["discourse"]},
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "application/json; charset=utf-8",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    ) as response:
        response_json = loads(await response.read())
        if response_json.get("message", None) is not None:
            retry_time = fromtimestamp(int(response.headers.get("X-RateLimit-Reset")))
            if retry_time > now():
                print(f"Retry time is {retry_time - now()}; waiting for 1 minute...")
                await sleep(60)
            return False
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--github_issue_number")
    parser.add_argument("--github_token")
    args = parser.parse_args()
    run(main(int(args.github_issue_number), args.github_token))
