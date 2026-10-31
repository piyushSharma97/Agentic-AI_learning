#!/usr/bin/env python
# coding: utf-8

"""
Logging: a record of what the program did, separate from print.

print is fine while you are learning a snippet. logging is what you keep
in an app, because you can turn levels on and off without deleting lines.

Levels, from quietest to loudest:
DEBUG, INFO, WARNING, ERROR, CRITICAL.

A logger shows a message when the message level is at least as high
as the logger's level. INFO hides DEBUG. ERROR hides INFO and WARNING.

JavaScript comparison:
- logging is in the Python standard library. JavaScript console.log, console.info, console.warn, and console.error are the everyday match.
- Named levels and a format string are closer to a library such as pino or winston than to console.log.
- logger.info("kept %s documents", documents) delays formatting. pino uses an object: logger.info({ documents }, "kept documents").
- logger.exception inside except prints the traceback. JavaScript console.error(error) prints the stack if you pass the error object.
"""

import logging

# --------------------------------------------------
# 1. Configure the root logger once
# --------------------------------------------------
# format puts the level and the logger name in front of your message.
# level=DEBUG shows every line in this lesson. A running app often uses INFO.

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)

# --------------------------------------------------
# 2. Get a named logger
# --------------------------------------------------
# __name__ is the module name. In a bigger app that tells you which file spoke.

logger = logging.getLogger("agent.lesson")

logger.debug("chunk splitter settings loaded")
logger.info("retriever returned 3 documents")
logger.warning("prompt is close to the token limit")
logger.error("model request failed")

# --------------------------------------------------
# 3. Pass data as extra arguments, not with an f-string
# --------------------------------------------------
# The logger formats the message only if the line will actually be emitted.
# %s is the logging style. It is not the same as an f-string.

documents = 3
logger.info("kept %s documents after the score filter", documents)

# --------------------------------------------------
# 4. exception() adds the traceback
# --------------------------------------------------
# Call it inside an except block. The message plus the stack are one log record.

def parse_score(text):
    return int(text)


try:
    parse_score("high")
except ValueError:
    logger.exception("could not parse the score")
