const state = {
  pipeline: [],
  run: null,
  stageIndex: 0,

  stageStartedAt: null,
  stageDurations: {},
  totalCompletedMs: 0,
  timerHandle: null,

  visualization: {
    stageId: null,
    items: [],
    index: 0,
  },

  /*
   * Final visualization data for each completed module.
   * Cleared ONLY by reset/new run.
   */
  visualizationCache: {},

  /*
   * Current example index for each visualization.
   */
  visualizationIndex: {},
};

const $ = (id) => document.getElementById(id);


/* ============================================================
   API
   ============================================================ */

async function api(url, options = {}) {
  const res = await fetch(url, options);

  let data;

  try {
    data = await res.json();
  } catch (_) {
    data = {};
  }

  if (!res.ok) {
    throw new Error(
      data.detail ||
      data.message ||
      `Request failed (${res.status})`
    );
  }

  return data;
}


/* ============================================================
   TIME
   ============================================================ */

function formatDuration(ms) {
  const totalSeconds =
    Math.max(0, Math.floor(ms / 1000));

  const hours =
    Math.floor(totalSeconds / 3600);

  const minutes =
    Math.floor(
      (totalSeconds % 3600) / 60
    );

  const seconds =
    totalSeconds % 60;

  if (hours > 0) {
    return (
      `${String(hours).padStart(2, '0')}:` +
      `${String(minutes).padStart(2, '0')}:` +
      `${String(seconds).padStart(2, '0')}`
    );
  }

  return (
    `${String(minutes).padStart(2, '0')}:` +
    `${String(seconds).padStart(2, '0')}`
  );
}

function clearTimer() {
  if (state.timerHandle) {
    clearInterval(state.timerHandle);
    state.timerHandle = null;
  }
}

function startStageTimer(stageId) {
  clearTimer();

  state.stageStartedAt =
    performance.now();

  const tick = () => {
    if (
      state.stageStartedAt === null
    ) {
      return;
    }

    const elapsed =
      performance.now() -
      state.stageStartedAt;

    $('stageTimer').textContent =
      formatDuration(elapsed);
  };

  tick();

  state.timerHandle =
    setInterval(
      tick,
      250
    );
}

function finishStageTimer(stageId) {
  if (
    state.stageStartedAt === null
  ) {
    return;
  }

  const elapsed =
    performance.now() -
    state.stageStartedAt;

  state.stageDurations[stageId] =
    elapsed;

  state.totalCompletedMs =
    Object.values(
      state.stageDurations
    ).reduce(
      (sum, value) =>
        sum + value,
      0
    );

  clearTimer();

  state.stageStartedAt =
    null;

  $('stageTimer').textContent =
    formatDuration(elapsed);
}


/* ============================================================
   CURRENT STAGE
   ============================================================ */

function currentStage() {
  return (
    state.pipeline[
      state.stageIndex
    ] || null
  );
}

function currentStageInfo() {
  const stage =
    currentStage();

  if (
    !stage ||
    !state.run
  ) {
    return null;
  }

  return (
    state.run.stages?.[
      stage.id
    ] || null
  );
}


/* ============================================================
   PIPELINE STATE
   ============================================================ */

function isPipelineRunning() {
  if (
    !state.run?.stages
  ) {
    return false;
  }

  return Object.values(
    state.run.stages
  ).some(
    stage =>
      stage?.status ===
      'running'
  );
}

function getLastCompletedIndex() {
  let last = -1;

  for (
    let i = 0;
    i < state.pipeline.length;
    i++
  ) {
    const stage =
      state.pipeline[i];

    const info =
      state.run?.stages?.[
        stage.id
      ];

    if (
      info?.status ===
      'complete'
    ) {
      last = i;
    } else {
      break;
    }
  }

  return last;
}


/* ============================================================
   SIDEBAR NAVIGATION
   ============================================================ */

function renderSteps() {
  const pipelineRunning =
    isPipelineRunning();

  const lastCompletedIndex =
    getLastCompletedIndex();

  $('stepList').classList.toggle(
    'navigation-locked',
    pipelineRunning
  );

  $('stepList').innerHTML =
    state.pipeline
      .map((stage, index) => {

        const status =
          state.run?.stages?.[
            stage.id
          ]?.status ||
          (
            index === 0
              ? 'ready'
              : 'locked'
          );

        const active =
          !!state.run &&
          index ===
            state.stageIndex;

        /*
         * Navigation rules:
         *
         * - While running: nothing clickable.
         * - Completed stages: clickable for review.
         * - Current unfinished stage: clickable.
         * - Locked future stages: not clickable.
         */
        const canOpen =
          !pipelineRunning &&
          (
            status === 'complete' ||
            index ===
              state.stageIndex
          );

        const historical =
          status === 'complete' &&
          index <
            lastCompletedIndex;

        return `
          <div
            class="
              step
              ${active ? 'active' : ''}
              ${status === 'complete' ? 'done' : ''}
              ${status === 'locked' ? 'locked' : ''}
              ${historical ? 'historical' : ''}
              ${canOpen ? 'clickable' : ''}
            "
            data-stage-index="${index}"
            role="${canOpen ? 'button' : 'presentation'}"
            ${canOpen ? 'tabindex="0"' : ''}
            aria-current="${active ? 'step' : 'false'}"
          >

            <div class="step-index">
              ${
                status === 'complete'
                  ? '✓'
                  : stage.number
              }
            </div>

            <div>
              <div class="step-title">
                ${escapeHtml(stage.title)}
              </div>

              <div class="step-sub">
                ${escapeHtml(stage.short)}
              </div>
            </div>

            <div class="check">
              ${
                status === 'complete'
                  ? 'saved'
                  : ''
              }
            </div>

          </div>
        `;
      })
      .join('');

  if (pipelineRunning) {
    return;
  }

  $('stepList')
    .querySelectorAll(
      '.step.clickable'
    )
    .forEach(step => {

      const open = () => {

        const index =
          Number(
            step.dataset.stageIndex
          );

        if (
          !Number.isInteger(index)
        ) {
          return;
        }

        openStageFromSidebar(index);
      };

      step.addEventListener(
        'click',
        open
      );

      step.addEventListener(
        'keydown',
        event => {

          if (
            event.key !== 'Enter' &&
            event.key !== ' '
          ) {
            return;
          }

          event.preventDefault();

          open();
        }
      );
    });
}

function openStageFromSidebar(index) {
  if (!state.run) {
    return;
  }

  if (isPipelineRunning()) {
    return;
  }

  if (
    index < 0 ||
    index >=
      state.pipeline.length
  ) {
    return;
  }

  const stage =
    state.pipeline[index];

  const info =
    state.run.stages?.[
      stage.id
    ];

  if (!info) {
    return;
  }

  const isCompleted =
    info.status ===
    'complete';

  const isCurrent =
    index ===
    state.stageIndex;

  if (
    !isCompleted &&
    !isCurrent
  ) {
    return;
  }

  state.stageIndex =
    index;

  showStage();
}


/* ============================================================
   RESET / START VIEW
   ============================================================ */

function showStartView() {
  clearTimer();

  state.run = null;

  state.stageIndex = 0;

  state.stageStartedAt =
    null;

  state.stageDurations = {};

  state.totalCompletedMs =
    0;

  /*
   * Reset is the ONLY operation that
   * clears visualization history.
   */
  state.visualization = {
    stageId: null,
    items: [],
    index: 0,
  };

  state.visualizationCache = {};
  state.visualizationIndex = {};

  $('startView')
    .classList.remove(
      'hidden'
    );

  $('stageView')
    .classList.add(
      'hidden'
    );

  $('totalTimeCard')
    .classList.add(
      'hidden'
    );

  $('fileList').innerHTML =
    '';

  $('pdfInput').value =
    '';

  $('metadataInput').value =
    '';

  $('runBtn')
    .classList.remove(
      'hidden'
    );

  $('nextBtn')
    .classList.add(
      'hidden'
    );

  $('resetBtn')
    .classList.add(
      'hidden'
    );

  clearUploadValidation();

  renderSteps();
}


/* ============================================================
   STATS
   ============================================================ */

function renderStats(stageId) {
  const info =
    state.run?.stages?.[
      stageId
    ];

  const output =
    info?.output || {};

  const items = [];

  if (
    output.rows !==
    undefined
  ) {
    items.push([
      'Rows',
      output.rows,
    ]);
  }

  if (
    output.entities !==
    undefined
  ) {
    items.push([
      'Entities',
      output.entities,
    ]);
  }

  if (
    output.relations !==
    undefined
  ) {
    items.push([
      'Relations',
      output.relations,
    ]);
  }

  if (
    output.triples !==
    undefined
  ) {
    items.push([
      'RDF triples',
      output.triples,
    ]);
  }

  $('stageStats').innerHTML =
    items
      .slice(0, 3)
      .map(
        ([key, value]) => `
          <div class="stat">
            <span>
              ${escapeHtml(key)}
            </span>

            <strong>
              ${escapeHtml(value)}
            </strong>
          </div>
        `
      )
      .join('');
}


/* ============================================================
   VISUALIZATION STAGE HELPERS
   ============================================================ */

function normalizeRelationStageId(stageId) {
  return String(stageId || '')
    .toLowerCase()
    .replace(
      /[\s-]+/g,
      '_'
    );
}

function isRelationExtractionStage(stageId) {
  return (
    normalizeRelationStageId(
      stageId
    ) ===
    'relation_extraction'
  );
}


/* ============================================================
   VISUALIZATION LOADING
   ============================================================ */

async function renderPreview(stageId) {
  const relationStage =
    isRelationExtractionStage(
      stageId
    );

  const normalVisualizationStages = [
    'entity-extraction',
    'entity-linking',
  ];

  const isVisualizationStage =
    relationStage ||
    normalVisualizationStages.includes(
      stageId
    );

  /*
   * No visualization for the other modules.
   */
  if (!isVisualizationStage) {
    $('preview')
      .classList.add(
        'hidden'
      );

    $('preview').innerHTML =
      '';

    return;
  }


  /* ----------------------------------------------------------
     REUSE CACHED RELATION VISUALIZATION
     ---------------------------------------------------------- */

  if (relationStage) {
    const cached =
      state.visualizationCache[
        stageId
      ];

    if (
      cached !== undefined
    ) {
      const savedIndex =
        state.visualizationIndex[
          stageId
        ] ?? 0;

      state.visualization = {
        stageId,
        items: cached,
        index: Math.min(
          savedIndex,
          Math.max(
            cached.length - 1,
            0
          )
        ),
      };

      renderRelationExtractionExample(
        stageId
      );

      return;
    }
  }


  /* ----------------------------------------------------------
     REUSE ENTITY VISUALIZATION CACHE
     ---------------------------------------------------------- */

  if (!relationStage) {
    const cached =
      state.visualizationCache[
        stageId
      ];

    if (
      cached !== undefined
    ) {
      const savedIndex =
        state.visualizationIndex[
          stageId
        ] ?? 0;

      state.visualization = {
        stageId,
        items: cached,
        index: Math.min(
          savedIndex,
          Math.max(
            cached.length - 1,
            0
          )
        ),
      };

      $('preview')
        .classList.remove(
          'hidden'
        );

      if (
        !cached.length
      ) {
        renderEmptyVisualization(
          stageId
        );

        return;
      }

      if (
        stageId ===
        'entity-extraction'
      ) {
        renderEntityExtractionExample();
      } else {
        renderEntityLinkingExample();
      }

      return;
    }
  }


  /* ----------------------------------------------------------
     FIND OUTPUT ARTIFACT
     ---------------------------------------------------------- */

  const files =
    state.run
      ?.artifacts?.[
        stageId
      ] || [];

  if (!files.length) {
    $('preview')
      .classList.add(
        'hidden'
      );

    $('preview').innerHTML =
      '';

    return;
  }

  const file =
    files.find(
      name =>
        String(name)
          .toLowerCase()
          .endsWith('.jsonl')
    ) ||
    files[0];


  /* ----------------------------------------------------------
     LOAD OUTPUT ARTIFACT
     ---------------------------------------------------------- */

  try {
    const res =
      await fetch(
        `/api/runs/${encodeURIComponent(
          state.run.run_id
        )}/artifacts/${encodeURIComponent(
          file
        )}`
      );

    if (!res.ok) {
      throw new Error(
        'Could not load visualization data.'
      );
    }

    const text =
      await res.text();

    const data =
      text
        .split(/\r?\n/)
        .map(
          line =>
            line.trim()
        )
        .filter(Boolean)
        .map(
          line =>
            JSON.parse(line)
        );


    /* --------------------------------------------------------
       RELATION EXTRACTION
       -------------------------------------------------------- */

    if (relationStage) {
      renderRelationExtractionVisualization(
        data,
        stageId
      );

      return;
    }


    /* --------------------------------------------------------
       ENTITY EXTRACTION
       -------------------------------------------------------- */

    if (
      stageId ===
      'entity-extraction'
    ) {
      renderEntityExtractionVisualization(
        data
      );

      return;
    }


    /* --------------------------------------------------------
       ENTITY LINKING
       -------------------------------------------------------- */

    if (
      stageId ===
      'entity-linking'
    ) {
      renderEntityLinkingVisualization(
        data
      );

      return;
    }

  } catch (err) {
    $('preview')
      .classList.remove(
        'hidden'
      );

    $('preview').innerHTML = `
      <div class="preview-title">
        Visualization unavailable
      </div>

      <div class="preview-row">
        ${escapeHtml(
          err.message
        )}
      </div>
    `;
  }
}

function renderEmptyVisualization(stageId) {
  $('preview')
    .classList.remove(
      'hidden'
    );

  if (
    stageId ===
    'entity-linking'
  ) {
    $('preview').innerHTML = `
      <div class="visualization-empty">
        No METHOD entities were found in this stage.
      </div>
    `;
  } else if (
    isRelationExtractionStage(
      stageId
    )
  ) {
    $('preview').innerHTML = `
      <div class="visualization-empty">
        No relations were extracted from this stage.
      </div>
    `;
  } else {
    $('preview').innerHTML = `
      <div class="visualization-empty">
        No entity-extraction examples were found in this stage.
      </div>
    `;
  }
}


/* ============================================================
   ENTITY EXTRACTION VISUALIZATION
   ============================================================ */

function renderEntityExtractionVisualization(
  data
) {
  const items =
    data.filter(
      item =>
        item &&
        typeof item.text ===
          'string'
    );

  state.visualizationCache[
    'entity-extraction'
  ] = items;

  const savedIndex =
    state.visualizationIndex[
      'entity-extraction'
    ] ?? 0;

  state.visualization = {
    stageId:
      'entity-extraction',

    items,

    index:
      Math.min(
        savedIndex,
        Math.max(
          items.length - 1,
          0
        )
      ),
  };

  $('preview')
    .classList.remove(
      'hidden'
    );

  if (!items.length) {
    renderEmptyVisualization(
      'entity-extraction'
    );

    return;
  }

  renderEntityExtractionExample();
}

function renderEntityExtractionExample() {
  const {
    items,
    index,
  } =
    state.visualization;

  if (!items.length) {
    return;
  }

  const item =
    items[index];

  if (!item) {
    return;
  }

  const spans =
    [...(
      item.spans || []
    )]
      .filter(
        span =>
          Number.isFinite(
            Number(
              span.start
            )
          ) &&
          Number.isFinite(
            Number(
              span.end
            )
          ) &&
          Number(
            span.end
          ) >
            Number(
              span.start
            )
      )
      .sort(
        (a, b) => {

          if (
            Number(
              a.start
            ) !==
            Number(
              b.start
            )
          ) {
            return (
              Number(a.start) -
              Number(b.start)
            );
          }

          return (
            Number(b.end) -
            Number(a.end)
          );
        }
      );

  const highlightedText =
    buildHighlightedText(
      item.text,
      spans
    );

  $('preview').innerHTML = `
    <div class="visualization-header">

      <div>

        <div class="preview-title">
          Entity Extraction
        </div>

        <div class="visualization-subtitle">
          METHOD · ACTIVITY · GOAL
        </div>

      </div>

      <div class="visualization-counter">
        ${index + 1} / ${items.length}
      </div>

    </div>

    <div class="entity-legend">

      <span>
        <i class="legend-dot method"></i>
        METHOD
      </span>

      <span>
        <i class="legend-dot activity"></i>
        ACTIVITY
      </span>

      <span>
        <i class="legend-dot goal"></i>
        GOAL
      </span>

    </div>

    <div class="ner-text">
      ${highlightedText}
    </div>

    <div class="visualization-controls">

      <button
        class="viz-btn"
        data-viz-control="first"
        type="button"
        aria-label="First example"
        title="First example"
      >
        ↺
      </button>

      <button
        class="viz-btn"
        data-viz-control="prev"
        type="button"
        aria-label="Previous example"
        title="Previous example"
      >
        ←
      </button>

      <button
        class="viz-btn"
        data-viz-control="next"
        type="button"
        aria-label="Next example"
        title="Next example"
      >
        →
      </button>

    </div>
  `;

  attachVisualizationControls(
    renderEntityExtractionExample
  );
}


/* ============================================================
   ENTITY LINKING VISUALIZATION
   ============================================================ */

function renderEntityLinkingVisualization(
  data
) {
  const examples = [];

  for (
    const item of data
  ) {
    const text =
      item?.text;

    if (!text) {
      continue;
    }

    for (
      const span of
        item.spans || []
    ) {
      if (
        span?.label !==
        'METHOD'
      ) {
        continue;
      }

      const start =
        Number(
          span.start
        );

      const end =
        Number(
          span.end
        );

      if (
        !Number.isInteger(
          start
        ) ||
        !Number.isInteger(
          end
        ) ||
        start < 0 ||
        end <= start ||
        end > text.length
      ) {
        continue;
      }

      const mention =
        text.slice(
          start,
          end
        );

      /*
       * Never display literal "None".
       *
       * Priority:
       *   1. proper_name
       *   2. wikipedia_title
       *   3. original mention
       */
      const rawProperName =
        span.proper_name;

      const rawWikipediaTitle =
        span.wikipedia_title;

      const properName =
        rawProperName &&
        rawProperName !==
          'None'
          ? rawProperName
          : (
              rawWikipediaTitle &&
              rawWikipediaTitle !==
                'None'
            )
              ? rawWikipediaTitle
              : mention;

      const wikipediaUrl =
        span.wikipedia_url &&
        span.wikipedia_url !==
          'None'
          ? span.wikipedia_url
          : null;

      examples.push({
        text,
        start,
        end,
        label: 'METHOD',
        properName,
        wikipediaUrl,
      });
    }
  }

  state.visualizationCache[
    'entity-linking'
  ] = examples;

  const savedIndex =
    state.visualizationIndex[
      'entity-linking'
    ] ?? 0;

  state.visualization = {
    stageId:
      'entity-linking',

    items:
      examples,

    index:
      Math.min(
        savedIndex,
        Math.max(
          examples.length - 1,
          0
        )
      ),
  };

  $('preview')
    .classList.remove(
      'hidden'
    );

  if (!examples.length) {
    renderEmptyVisualization(
      'entity-linking'
    );

    return;
  }

  renderEntityLinkingExample();
}

function renderEntityLinkingExample() {
  const {
    items,
    index,
  } =
    state.visualization;

  if (!items.length) {
    return;
  }

  const item =
    items[index];

  if (!item) {
    return;
  }

  const highlightedText =
    buildHighlightedText(
      item.text,
      [
        {
          start:
            item.start,

          end:
            item.end,

          label:
            'METHOD',
        },
      ]
    );

  const linkHtml =
    item.wikipediaUrl
      ? `
          <a
            class="entity-link"
            href="${escapeAttribute(
              item.wikipediaUrl
            )}"
            target="_blank"
            rel="noopener noreferrer"
          >
            Open Wikipedia ↗
          </a>
        `
      : `
          <span class="entity-link-disabled">
            No Wikipedia link
          </span>
        `;

  $('preview')
    .classList.remove(
      'hidden'
    );

  $('preview').innerHTML = `
    <div class="visualization-header">

      <div>

        <div class="preview-title">
          Entity Linking
        </div>

        <div class="visualization-subtitle">
          METHOD
        </div>

      </div>

      <div class="visualization-counter">
        ${index + 1} / ${items.length}
      </div>

    </div>

    <div class="ner-text linking-text">
      ${highlightedText}
    </div>

    <div class="entity-details">

      <div class="entity-detail-label">
        LINKED ENTITY
      </div>

      <div class="entity-name">
        ${escapeHtml(
          item.properName
        )}
      </div>

      ${linkHtml}

    </div>

    <div class="visualization-controls">

      <button
        class="viz-btn"
        data-viz-control="first"
        type="button"
        aria-label="First example"
        title="First example"
      >
        ↺
      </button>

      <button
        class="viz-btn"
        data-viz-control="prev"
        type="button"
        aria-label="Previous example"
        title="Previous example"
      >
        ←
      </button>

      <button
        class="viz-btn"
        data-viz-control="next"
        type="button"
        aria-label="Next example"
        title="Next example"
      >
        →
      </button>

    </div>
  `;

  attachVisualizationControls(
    renderEntityLinkingExample
  );
}


/* ============================================================
   RELATION EXTRACTION VISUALIZATION
   ============================================================ */

/*
 * The notebook creates one visualization example
 * for every:
 *
 *   sentence + relation
 *
 * We preserve that exact structure here, while handling
 * the important difference between Python character offsets
 * and JavaScript UTF-16 offsets.
 */

function relationKind(label) {
  const normalized =
    String(
      label || ''
    ).toUpperCase();

  if (
    normalized ===
    'EMPLOYS'
  ) {
    return 'employs';
  }

  if (
    normalized ===
    'HAS_OBJECTIVE'
  ) {
    return 'has-objective';
  }

  return 'relation';
}

function relationRangeKind(label) {
  const normalized =
    String(
      label || ''
    ).toUpperCase();

  if (
    normalized ===
    'METHOD'
  ) {
    return 'method';
  }

  if (
    normalized ===
    'GOAL'
  ) {
    return 'goal';
  }

  return 'generic';
}


/*
 * Build the browser-side relation visualization items.
 *
 * Backend NER offsets come from spaCy/Python and therefore use
 * Python string indexing (Unicode code points).
 */
function buildRelationVisualizationItems(
  data
) {
  const items = [];

  for (const row of data || []) {
    const text =
      String(
        row?.text ?? ''
      );

    if (!text) {
      continue;
    }

    /*
     * Python len(text) counts Unicode code points.
     */
    const pythonLength =
      Array.from(text).length;

    for (
      const relation of
        row?.relations || []
    ) {
      const domain =
        relation?.domain;

      const range =
        relation?.range;

      if (
        !domain ||
        !range
      ) {
        continue;
      }

      const domainStart =
        Number(
          domain.start
        );

      const domainEnd =
        Number(
          domain.end
        );

      const rangeStart =
        Number(
          range.start
        );

      const rangeEnd =
        Number(
          range.end
        );

      if (
        !Number.isInteger(
          domainStart
        ) ||
        !Number.isInteger(
          domainEnd
        ) ||
        !Number.isInteger(
          rangeStart
        ) ||
        !Number.isInteger(
          rangeEnd
        )
      ) {
        continue;
      }

      if (
        domainStart < 0 ||
        domainEnd <= domainStart ||
        domainEnd > pythonLength
      ) {
        continue;
      }

      if (
        rangeStart < 0 ||
        rangeEnd <= rangeStart ||
        rangeEnd > pythonLength
      ) {
        continue;
      }

      /*
       * Recreate the span using Python-style indexing.
       */
      const domainText =
        slicePythonText(
          text,
          domainStart,
          domainEnd
        );

      const rangeText =
        slicePythonText(
          text,
          rangeStart,
          rangeEnd
        );

      /*
       * Backend now provides exact span text.
       * Use that as an additional validation guard.
       */
      if (
        domain.text !== undefined &&
        domain.text !== domainText
      ) {
        console.warn(
          'Skipping relation with a domain offset/text mismatch.',
          relation
        );

        continue;
      }

      if (
        range.text !== undefined &&
        range.text !== rangeText
      ) {
        console.warn(
          'Skipping relation with a range offset/text mismatch.',
          relation
        );

        continue;
      }

      items.push({
        text,

        relation: {
          label:
            relation.label ||
            'RELATION',

          offsetUnit:
            relation.offset_unit ||
            'python_codepoint',

          domain: {
            start:
              domainStart,

            end:
              domainEnd,

            label:
              domain.label ||
              'ACTIVITY',

            text:
              domainText,
          },

          range: {
            start:
              rangeStart,

            end:
              rangeEnd,

            label:
              range.label ||
              'ENTITY',

            text:
              rangeText,
          },
        },
      });
    }
  }

  return items;
}


/*
 * Return a substring using Python-style indexing.
 *
 * Python:
 *   text[start:end]
 *
 * JavaScript:
 *   text.slice(start, end)
 *
 * are NOT equivalent when the string contains characters represented
 * by UTF-16 surrogate pairs.
 */
function slicePythonText(
  text,
  start,
  end
) {
  return Array.from(
    text
  )
    .slice(
      start,
      end
    )
    .join('');
}


/*
 * Convert a Python Unicode-code-point offset into the corresponding
 * JavaScript UTF-16 offset.
 *
 * Example:
 *
 *   Text: A😀B
 *
 * Python offsets:
 *   A = 0
 *   😀 = 1
 *   B = 2
 *
 * JavaScript UTF-16 offsets:
 *   A = 0
 *   😀 = 1..2
 *   B = 3
 */
function pythonOffsetToUtf16(
  text,
  pythonOffset
) {
  if (
    !Number.isInteger(
      pythonOffset
    ) ||
    pythonOffset < 0
  ) {
    return null;
  }

  let codePointIndex =
    0;

  let utf16Offset =
    0;

  for (
    const char of
      text
  ) {
    if (
      codePointIndex >=
      pythonOffset
    ) {
      break;
    }

    utf16Offset +=
      char.length;

    codePointIndex +=
      1;
  }

  if (
    codePointIndex !==
    pythonOffset
  ) {
    return null;
  }

  return utf16Offset;
}


/*
 * Create an SVG element safely.
 */
function createSvgElement(
  tag,
  attrs = {}
) {
  const element =
    document.createElementNS(
      'http://www.w3.org/2000/svg',
      tag
    );

  for (
    const [
      key,
      value,
    ] of Object.entries(
      attrs
    )
  ) {
    element.setAttribute(
      key,
      String(value)
    );
  }

  return element;
}


/*
 * Obtain the sentence text node.
 *
 * The renderer inserts the sentence through textContent, so there
 * is exactly one plain text node and the backend offsets continue
 * to refer to the original sentence.
 */
function getRelationTextNode(
  sentence
) {
  const textContainer =
    sentence?.querySelector(
      '.relation-text'
    );

  if (
    !textContainer
  ) {
    return null;
  }

  return (
    textContainer.firstChild &&
    textContainer.firstChild.nodeType ===
      Node.TEXT_NODE
      ? textContainer.firstChild
      : null
  );
}


/*
 * Convert Python code-point offsets into DOM Range offsets.
 *
 * Range APIs use JavaScript string indexing, which is UTF-16 based.
 */
function getTextRangeRects(
  textNode,
  start,
  end,
  container
) {
  if (
    !textNode ||
    !container
  ) {
    return [];
  }

  const text =
    textNode.data ||
    '';

  const pythonLength =
    Array.from(
      text
    ).length;

  if (
    start < 0 ||
    end <= start ||
    start >= pythonLength ||
    end > pythonLength
  ) {
    return [];
  }

  const jsStart =
    pythonOffsetToUtf16(
      text,
      start
    );

  const jsEnd =
    pythonOffsetToUtf16(
      text,
      end
    );

  if (
    jsStart === null ||
    jsEnd === null ||
    jsEnd <= jsStart
  ) {
    return [];
  }

  const range =
    document.createRange();

  range.setStart(
    textNode,
    jsStart
  );

  range.setEnd(
    textNode,
    jsEnd
  );

  const containerRect =
    container.getBoundingClientRect();

  return Array.from(
    range.getClientRects()
  )
    .filter(
      rect =>
        rect.width > 0 &&
        rect.height > 0
    )
    .map(
      rect => ({
        left:
          rect.left -
          containerRect.left,

        right:
          rect.right -
          containerRect.left,

        top:
          rect.top -
          containerRect.top,

        bottom:
          rect.bottom -
          containerRect.top,

        width:
          rect.width,

        height:
          rect.height,
      })
    );
}


/*
 * Choose the rectangles that should anchor the relation arrow.
 *
 * This is important when:
 *
 * 1. an entity wraps onto multiple visual lines;
 * 2. the two relation spans overlap, as EMPLOYS can do.
 */
function chooseRelationAnchorRects(
  domainRects,
  rangeRects,
  relation
) {
  const characterOverlap =
    relation.domain.start <
      relation.range.end &&
    relation.range.start <
      relation.domain.end;

  /*
   * Overlapping relation spans.
   *
   * EMPLOYS in the original notebook is generated specifically from
   * overlapping ACTIVITY/METHOD spans.
   */
  if (
    characterOverlap
  ) {
    let bestDomain =
      domainRects[0];

    let bestRange =
      rangeRects[0];

    let bestScore =
      -Infinity;

    for (
      const domainRect of
        domainRects
    ) {
      for (
        const rangeRect of
          rangeRects
      ) {
        const overlapWidth =
          Math.max(
            0,
            Math.min(
              domainRect.right,
              rangeRect.right
            ) -
              Math.max(
                domainRect.left,
                rangeRect.left
              )
          );

        const overlapHeight =
          Math.max(
            0,
            Math.min(
              domainRect.bottom,
              rangeRect.bottom
            ) -
              Math.max(
                domainRect.top,
                rangeRect.top
              )
          );

        const overlapArea =
          overlapWidth *
          overlapHeight;

        const centerDistance =
          Math.abs(
            (
              domainRect.top +
              domainRect.bottom
            ) / 2 -
              (
                rangeRect.top +
                rangeRect.bottom
              ) / 2
          ) +

          Math.abs(
            (
              domainRect.left +
              domainRect.right
            ) / 2 -
              (
                rangeRect.left +
                rangeRect.right
              ) / 2
          ) * 0.05;

        const score =
          overlapArea * 1000 -
          centerDistance;

        if (
          score >
          bestScore
        ) {
          bestScore =
            score;

          bestDomain =
            domainRect;

          bestRange =
            rangeRect;
        }
      }
    }

    return {
      domainRect:
        bestDomain,

      rangeRect:
        bestRange,
    };
  }

  /*
   * Normal relation where domain appears first.
   *
   * Use the LAST rectangle of the domain and the FIRST rectangle
   * of the range. This is much more reliable when either entity
   * wraps onto multiple lines.
   */
  if (
    relation.domain.end <=
    relation.range.start
  ) {
    return {
      domainRect:
        domainRects[
          domainRects.length - 1
        ],

      rangeRect:
        rangeRects[0],
    };
  }

  /*
   * Range appears first.
   *
   * Domain remains the source of the arrow, so use the FIRST
   * domain rectangle and LAST range rectangle.
   */
  return {
    domainRect:
      domainRects[0],

    rangeRect:
      rangeRects[
        rangeRects.length - 1
      ],
  };
}


/*
 * Draw the current relation.
 */
function drawRelationVisualization(
  root
) {
  if (!root) {
    return;
  }

  const sentence =
    root.querySelector(
      '.relation-sentence'
    );

  const textNode =
    getRelationTextNode(
      sentence
    );

  const svg =
    root.querySelector(
      '.relation-svg'
    );

  if (
    !sentence ||
    !textNode ||
    !svg
  ) {
    return;
  }

  const item =
    root._relationItem;

  if (
    !item?.relation
  ) {
    return;
  }

  const relation =
    item.relation;

  /*
   * Resolve the exact backend spans.
   */
  const domainRects =
    getTextRangeRects(
      textNode,
      relation.domain.start,
      relation.domain.end,
      sentence
    );

  const rangeRects =
    getTextRangeRects(
      textNode,
      relation.range.start,
      relation.range.end,
      sentence
    );

  if (
    !domainRects.length ||
    !rangeRects.length
  ) {
    console.warn(
      'Could not resolve relation rectangles.',
      relation
    );

    return;
  }

  /*
   * Remove the old drawing before recalculating it.
   */
  svg.replaceChildren();

  sentence
    .querySelectorAll(
      '.relation-highlight'
    )
    .forEach(
      node =>
        node.remove()
    );

  sentence
    .querySelectorAll(
      '.relation-edge-label'
    )
    .forEach(
      node =>
        node.remove()
    );


  /*
   * ----------------------------------------------------------
   * ENTITY HIGHLIGHTS
   * ----------------------------------------------------------
   */

  function addHighlight(
    rects,
    className,
    label
  ) {
    rects.forEach(
      (
        rect,
        index
      ) => {
        const box =
          document.createElement(
            'div'
          );

        box.className =
          `relation-highlight ${className}`;

        box.style.left =
          `${rect.left - 2}px`;

        box.style.top =
          `${rect.top - 2}px`;

        box.style.width =
          `${rect.width + 4}px`;

        box.style.height =
          `${rect.height + 4}px`;

        /*
         * Only put the semantic label on the first visual line.
         */
        if (
          index === 0 &&
          label
        ) {
          const labelNode =
            document.createElement(
              'span'
            );

          labelNode.className =
            'relation-entity-label';

          labelNode.textContent =
            label;

          box.appendChild(
            labelNode
          );
        }

        sentence.appendChild(
          box
        );
      }
    );
  }

  addHighlight(
    domainRects,
    'domain',
    relation.domain.label ||
      'ACTIVITY'
  );

  addHighlight(
    rangeRects,
    `range ${relationRangeKind(
      relation.range.label
    )}`,
    relation.range.label ||
      'ENTITY'
  );


  /*
   * ----------------------------------------------------------
   * SVG SIZE
   * ----------------------------------------------------------
   */

  const width =
    sentence.clientWidth;

  const height =
    sentence.clientHeight;

  svg.setAttribute(
    'viewBox',
    `0 0 ${width} ${height}`
  );

  svg.setAttribute(
    'width',
    width
  );

  svg.setAttribute(
    'height',
    height
  );


  /*
   * ----------------------------------------------------------
   * ARROW MARKER
   * ----------------------------------------------------------
   */

  const defs =
    createSvgElement(
      'defs'
    );

  /*
   * Unique marker ID prevents collisions when the visualization
   * is redrawn.
   */
  const markerId =
    `relation-arrow-marker-${Math.random()
      .toString(36)
      .slice(2)}`;

  const marker =
    createSvgElement(
      'marker',
      {
        id:
          markerId,

        viewBox:
          '0 0 10 10',

        refX:
          '9',

        refY:
          '5',

        markerWidth:
          '7',

        markerHeight:
          '7',

        orient:
          'auto',
      }
    );

  const markerPath =
    createSvgElement(
      'path',
      {
        d:
          'M 0 0 L 10 5 L 0 10 z',
      }
    );

  markerPath.style.fill =
    'var(--accent)';

  marker.appendChild(
    markerPath
  );

  defs.appendChild(
    marker
  );

  svg.appendChild(
    defs
  );


  /*
   * ----------------------------------------------------------
   * CHOOSE ARROW ANCHORS
   * ----------------------------------------------------------
   */

  const {
    domainRect,
    rangeRect,
  } =
    chooseRelationAnchorRects(
      domainRects,
      rangeRects,
      relation
    );

  const overlaps =
    relation.domain.start <
      relation.range.end &&
    relation.range.start <
      relation.domain.end;

  let sx;
  let sy;
  let tx;
  let ty;


  /*
   * ----------------------------------------------------------
   * OVERLAPPING RELATION
   * ----------------------------------------------------------
   */

  if (
    overlaps
  ) {
    /*
     * Preserve original notebook semantics for EMPLOYS:
     * the domain and range can overlap.
     *
     * Put the endpoints on opposite sides so the arrow
     * remains visible rather than collapsing inside the
     * same highlighted area.
     */
    sx =
      domainRect.left - 4;

    sy =
      domainRect.top - 4;

    tx =
      rangeRect.right + 4;

    ty =
      rangeRect.top - 4;


  /*
   * ----------------------------------------------------------
   * DOMAIN BEFORE RANGE
   * ----------------------------------------------------------
   */

  } else if (
    relation.domain.end <=
    relation.range.start
  ) {
    sx =
      domainRect.right + 2;

    sy =
      domainRect.top - 4;

    tx =
      rangeRect.left - 2;

    ty =
      rangeRect.top - 4;


  /*
   * ----------------------------------------------------------
   * RANGE BEFORE DOMAIN
   * ----------------------------------------------------------
   */

  } else {
    /*
     * Relation direction is still:
     *
     *   domain -> range
     */
    sx =
      domainRect.left - 2;

    sy =
      domainRect.top - 4;

    tx =
      rangeRect.right + 2;

    ty =
      rangeRect.top - 4;
  }


  /*
   * ----------------------------------------------------------
   * CURVE LANE
   * ----------------------------------------------------------
   */

  const highestEntityTop =
    Math.min(
      sy,
      ty
    );

  let laneY =
    highestEntityTop - 55;

  laneY =
    Math.max(
      18,
      laneY
    );

  laneY =
    Math.min(
      laneY,

      Math.max(
        18,
        height - 30
      )
    );


  /*
   * ----------------------------------------------------------
   * CURVED DOMAIN -> RANGE ARROW
   * ----------------------------------------------------------
   */

  const pathData =
    [
      `M ${sx} ${sy}`,

      `C ${sx} ${laneY}`,

      `${tx} ${laneY}`,

      `${tx} ${ty}`,
    ].join(
      ' '
    );

  const path =
    createSvgElement(
      'path',
      {
        d:
          pathData,

        fill:
          'none',

        stroke:
          'var(--accent)',

        'stroke-width':
          '2.5',

        'stroke-linecap':
          'round',

        'stroke-linejoin':
          'round',

        'marker-end':
          `url(#${markerId})`,
      }
    );

  svg.appendChild(
    path
  );


  /*
   * ----------------------------------------------------------
   * RELATION LABEL
   * ----------------------------------------------------------
   */

  const edgeLabel =
    document.createElement(
      'div'
    );

  const kind =
    relationKind(
      relation.label
    );

  edgeLabel.className =
    `relation-edge-label ${kind}`;

  edgeLabel.textContent =
    relation.label ||
    'RELATION';

  const labelX =
    (
      sx +
      tx
    ) / 2;

  const labelY =
    (
      0.125 * sy +
      0.75 * laneY +
      0.125 * ty
    ) - 4;

  edgeLabel.style.left =
    `${labelX}px`;

  edgeLabel.style.top =
    `${Math.max(
      15,
      labelY
    )}px`;

  sentence.appendChild(
    edgeLabel
  );
}


/*
 * Render relation visualization from JSONL.
 */
function renderRelationExtractionVisualization(
  data,
  stageId
) {
  const items =
    buildRelationVisualizationItems(
      data
    );

  state.visualizationCache[
    stageId
  ] =
    items;

  const savedIndex =
    state.visualizationIndex[
      stageId
    ] ?? 0;

  state.visualizationIndex[
    stageId
  ] =
    Math.min(
      savedIndex,

      Math.max(
        items.length - 1,
        0
      )
    );

  state.visualization = {
    stageId,

    items,

    index:
      state.visualizationIndex[
        stageId
      ],
  };

  $('preview')
    .classList.remove(
      'hidden'
    );

  if (
    !items.length
  ) {
    renderEmptyVisualization(
      stageId
    );

    return;
  }

  renderRelationExtractionExample(
    stageId
  );
}


/*
 * Render the selected relation example.
 */
function renderRelationExtractionExample(
  stageId
) {
  const items =
    state.visualizationCache[
      stageId
    ] || [];

  const index =
    state.visualizationIndex[
      stageId
    ] ?? 0;

  if (
    !items.length
  ) {
    renderEmptyVisualization(
      stageId
    );

    return;
  }

  const safeIndex =
    Math.max(
      0,

      Math.min(
        index,
        items.length - 1
      )
    );

  state.visualization = {
    stageId,

    items,

    index:
      safeIndex,
  };

  const item =
    items[
      safeIndex
    ];

  if (!item) {
    return;
  }

  const relation =
    item.relation;

  const kind =
    relationKind(
      relation.label
    );


  /*
   * IMPORTANT:
   *
   * Do NOT inject item.text with innerHTML.
   *
   * We first create an empty .relation-text container,
   * then insert the sentence with textContent.
   *
   * This guarantees that the browser gets exactly the same
   * character sequence as the backend.
   */
  $('preview').innerHTML = `
    <div class="relation-visualization">

      <div class="relation-header">

        <div>
          <div class="preview-title">
            Relation Extraction
          </div>

          <div class="visualization-subtitle">
            ${escapeHtml(
              relation.domain.label ||
              'ACTIVITY'
            )}
            →
            ${escapeHtml(
              relation.range.label ||
              'ENTITY'
            )}
          </div>
        </div>

        <div>
          <span
            class="relation-type-pill ${kind}"
          >
            ${escapeHtml(
              relation.label ||
              'RELATION'
            )}
          </span>

          <div
            class="relation-counter"
            style="margin-top: 8px;"
          >
            ${safeIndex + 1}
            /
            ${items.length}
          </div>
        </div>

      </div>

      <div
        class="relation-sentence"
        id="relationSentence"
      >
        <div class="relation-text"></div>

        <svg
          class="relation-svg"
          aria-hidden="true"
        ></svg>
      </div>

      <div class="visualization-controls">

        <button
          class="viz-btn"
          data-viz-control="first"
          type="button"
          aria-label="First relation"
          title="First relation"
        >
          ↺
        </button>

        <button
          class="viz-btn"
          data-viz-control="prev"
          type="button"
          aria-label="Previous relation"
          title="Previous relation"
        >
          ←
        </button>

        <button
          class="viz-btn"
          data-viz-control="next"
          type="button"
          aria-label="Next relation"
          title="Next relation"
        >
          →
        </button>

      </div>

    </div>
  `;


  /*
   * Insert the exact sentence as plain text.
   */
  const relationText =
    $('preview')
      .querySelector(
        '.relation-text'
      );

  if (
    !relationText
  ) {
    return;
  }

  relationText.textContent =
    item.text;


  /*
   * Visualization controls.
   */
  attachVisualizationControls(
    () => {
      const count =
        state.visualizationCache[
          stageId
        ]?.length || 0;

      if (!count) {
        return;
      }

      state.visualizationIndex[
        stageId
      ] =
        state.visualization.index;

      renderRelationExtractionExample(
        stageId
      );
    },

    stageId
  );


  /*
   * Get visualization root.
   */
  const root =
    $('preview')
      .querySelector(
        '.relation-visualization'
      );

  if (
    !root
  ) {
    return;
  }

  root._relationItem =
    item;


  /*
   * Draw after layout exists.
   */
  requestAnimationFrame(
    () => {
      drawRelationVisualization(
        root
      );

      /*
       * Recalculate once fonts have loaded.
       */
      if (
        document.fonts?.ready
      ) {
        document.fonts.ready.then(
          () => {
            drawRelationVisualization(
              root
            );
          }
        );
      }
    }
  );
}


/* ============================================================
   RELATION RESIZE
   ============================================================ */

window.addEventListener(
  'resize',
  () => {

    const root =
      $('preview')
        ?.querySelector(
          '.relation-visualization'
        );

    if (root) {

      requestAnimationFrame(
        () => {
          drawRelationVisualization(
            root
          );
        }
      );
    }
  }
);


/* ============================================================
   TEXT HIGHLIGHTING
   ============================================================ */

function buildHighlightedText(
  text,
  spans
) {
  if (!text) {
    return '';
  }

  const validSpans =
    spans
      .map(
        span => ({
          ...span,

          start:
            Number(
              span.start
            ),

          end:
            Number(
              span.end
            ),
        })
      )
      .filter(
        span =>
          Number.isInteger(
            span.start
          ) &&
          Number.isInteger(
            span.end
          ) &&
          span.start >= 0 &&
          span.end >
            span.start &&
          span.end <=
            text.length
      )
      .sort(
        (a, b) => {

          if (
            a.start !==
            b.start
          ) {
            return (
              a.start -
              b.start
            );
          }

          return (
            b.end -
            a.end
          );
        }
      );

  let cursor = 0;
  let html = '';

  for (
    const span of
      validSpans
  ) {

    /*
     * Ignore overlapping spans in the
     * normal Entity Extraction visualization.
     *
     * Relation Extraction does NOT use this
     * function, so overlapping relations
     * remain fully supported there.
     */
    if (
      span.start <
      cursor
    ) {
      continue;
    }

    const before =
      text.slice(
        cursor,
        span.start
      );

    const entityText =
      text.slice(
        span.start,
        span.end
      );

    html += escapeHtml(
      collapseDisplayWhitespace(
        before
      )
    );

    const label =
      span.label ||
      'METHOD';

    const labelClass =
      label.toLowerCase();

    html += `
      <span
        class="entity-span ${escapeAttribute(
          labelClass
        )}"
      >

        <span class="entity-text">
          ${escapeHtml(
            collapseDisplayWhitespace(
              entityText
            )
          )}
        </span>

        <span class="entity-label">
          ${escapeHtml(
            label
          )}
        </span>

      </span>
    `;

    cursor =
      span.end;
  }

  html += escapeHtml(
    collapseDisplayWhitespace(
      text.slice(cursor)
    )
  );

  return html;
}

function collapseDisplayWhitespace(
  value
) {
  return String(value)
    .replace(
      /[\t\r\n\u00A0 ]+/g,
      ' '
    )
    .replace(
      /^ +| +$/g,
      ' '
    );
}


/* ============================================================
   VISUALIZATION CONTROLS
   ============================================================ */

function attachVisualizationControls(
  renderFunction,
  stageIdOverride = null
) {
  const first =
    $('preview')
      .querySelector(
        '[data-viz-control="first"]'
      );

  const prev =
    $('preview')
      .querySelector(
        '[data-viz-control="prev"]'
      );

  const next =
    $('preview')
      .querySelector(
        '[data-viz-control="next"]'
      );

  const getStageId =
    () =>
      stageIdOverride ||
      state.visualization.stageId;


  first?.addEventListener(
    'click',
    () => {

      const stageId =
        getStageId();

      state.visualization.index =
        0;

      if (stageId) {
        state.visualizationIndex[
          stageId
        ] = 0;
      }

      renderFunction();
    }
  );

  prev?.addEventListener(
    'click',
    () => {

      const stageId =
        getStageId();

      const count =
        state.visualization.items
          .length;

      if (!count) {
        return;
      }

      state.visualization.index =
        (
          state.visualization.index -
          1 +
          count
        ) % count;

      if (stageId) {
        state.visualizationIndex[
          stageId
        ] =
          state.visualization.index;
      }

      renderFunction();
    }
  );

  next?.addEventListener(
    'click',
    () => {

      const stageId =
        getStageId();

      const count =
        state.visualization.items
          .length;

      if (!count) {
        return;
      }

      state.visualization.index =
        (
          state.visualization.index +
          1
        ) % count;

      if (stageId) {
        state.visualizationIndex[
          stageId
        ] =
          state.visualization.index;
      }

      renderFunction();
    }
  );
}


/* ============================================================
   ESCAPING
   ============================================================ */

function escapeAttribute(value) {
  return String(value)
    .replaceAll(
      '&',
      '&amp;'
    )
    .replaceAll(
      '"',
      '&quot;'
    )
    .replaceAll(
      '<',
      '&lt;'
    )
    .replaceAll(
      '>',
      '&gt;'
    );
}

function escapeHtml(str) {
  return String(str)
    .replaceAll(
      '&',
      '&amp;'
    )
    .replaceAll(
      '<',
      '&lt;'
    )
    .replaceAll(
      '>',
      '&gt;'
    )
    .replaceAll(
      '"',
      '&quot;'
    );
}


/* ============================================================
   THEME
   ============================================================ */

function applyTheme(theme) {
  document.documentElement.dataset.theme =
    theme;

  localStorage.setItem(
    'research-spotlight-theme',
    theme
  );

  $('themeIcon').textContent =
    theme === 'dark'
      ? '☀'
      : '☾';

  $('themeToggle').title =
    theme === 'dark'
      ? 'Switch to light theme'
      : 'Switch to dark theme';

  $('themeToggle').setAttribute(
    'aria-label',
    theme === 'dark'
      ? 'Switch to light theme'
      : 'Switch to dark theme'
  );
}

function initTheme() {
  const stored =
    localStorage.getItem(
      'research-spotlight-theme'
    );

  applyTheme(
    stored === 'light'
      ? 'light'
      : 'dark'
  );
}


/* ============================================================
   PDF / METADATA VALIDATION
   ============================================================ */

function getPdfId(filename) {
  const match =
    filename.match(
      /^(\d+)\.pdf$/i
    );

  if (!match) {
    return null;
  }

  return Number(
    match[1]
  );
}

async function readMetadataRecords(
  file
) {
  if (!file) {
    throw new Error(
      'Metadata file is required.'
    );
  }

  const text =
    await file.text();

  const lines =
    text
      .replace(
        /^\uFEFF/,
        ''
      )
      .split(/\r?\n/)
      .map(
        line =>
          line.trim()
      )
      .filter(Boolean);

  if (!lines.length) {
    throw new Error(
      'The metadata file is empty.'
    );
  }

  const records = [];

  for (
    let i = 0;
    i < lines.length;
    i++
  ) {
    try {
      const record =
        JSON.parse(
          lines[i]
        );

      if (
        record === null ||
        typeof record !==
          'object' ||
        Array.isArray(
          record
        )
      ) {
        throw new Error(
          'Each metadata line must contain a JSON object.'
        );
      }

      records.push(
        record
      );

    } catch (err) {
      throw new Error(
        `Invalid metadata JSON on line ${i + 1}: ${err.message}`
      );
    }
  }

  return records;
}

async function validatePdfMetadata(
  pdfFiles,
  metadataFile
) {
  const errors = [];

  if (
    !pdfFiles.length
  ) {
    errors.push(
      'Please select at least one PDF.'
    );

    return errors;
  }

  if (!metadataFile) {
    errors.push(
      'Metadata is required. Please select the metadata.jsonl file.'
    );

    return errors;
  }

  const pdfEntries = [];

  for (
    const file of
      pdfFiles
  ) {
    const id =
      getPdfId(
        file.name
      );

    if (id === null) {
      errors.push(
        `${file.name} is invalid. PDFs must be named N.pdf, for example 1.pdf or 12.pdf.`
      );

      continue;
    }

    pdfEntries.push({
      file,
      id,
    });
  }

  if (errors.length) {
    return errors;
  }

  let metadataRecords;

  try {
    metadataRecords =
      await readMetadataRecords(
        metadataFile
      );

  } catch (err) {

    errors.push(
      err.message
    );

    return errors;
  }

  const metadataIds =
    new Map();

  for (
    const record of
      metadataRecords
  ) {

    if (
      !record.meta ||
      typeof record.meta !==
        'object' ||
      Array.isArray(
        record.meta
      )
    ) {
      errors.push(
        'A metadata record is missing the required "meta" object.'
      );

      continue;
    }

    if (
      record.meta.id ===
        undefined ||
      record.meta.id ===
        null ||
      record.meta.id === ''
    ) {
      errors.push(
        'A metadata record is missing the required "meta.id" field.'
      );

      continue;
    }

    const id =
      Number(
        record.meta.id
      );

    if (
      !Number.isInteger(id)
    ) {
      errors.push(
        `Metadata contains an invalid meta.id value: ${record.meta.id}.`
      );

      continue;
    }

    if (
      metadataIds.has(id)
    ) {
      errors.push(
        `Metadata contains duplicate records for meta.id: ${id}.`
      );

      continue;
    }

    metadataIds.set(
      id,
      record
    );
  }


  /*
   * Every uploaded PDF must have
   * matching metadata.
   */
  for (
    const {
      file,
      id,
    } of pdfEntries
  ) {

    if (
      !metadataIds.has(id)
    ) {
      errors.push(
        `${file.name} requires a metadata record with "meta.id": "${id}".`
      );
    }
  }


  /*
   * Every metadata ID must have
   * a corresponding PDF.
   */
  const uploadedPdfIds =
    new Set(
      pdfEntries.map(
        entry =>
          entry.id
      )
    );

  for (
    const id of
      metadataIds.keys()
  ) {

    if (
      !uploadedPdfIds.has(id)
    ) {
      errors.push(
        `Metadata contains meta.id: "${id}", but ${id}.pdf was not uploaded.`
      );
    }
  }


  /*
   * Duplicate PDF IDs are not allowed.
   */
  const seenPdfIds =
    new Set();

  for (
    const {
      file,
      id,
    } of pdfEntries
  ) {

    if (
      seenPdfIds.has(id)
    ) {
      errors.push(
        `More than one PDF corresponds to id: ${id} (${file.name}).`
      );
    }

    seenPdfIds.add(id);
  }

  return errors;
}


/* ============================================================
   UPLOAD VALIDATION UI
   ============================================================ */

function getOrCreateUploadValidation() {
  let el =
    $('uploadValidation');

  if (el) {
    return el;
  }

  el =
    document.createElement(
      'div'
    );

  el.id =
    'uploadValidation';

  el.setAttribute(
    'role',
    'status'
  );

  el.style.marginTop =
    '12px';

  el.style.whiteSpace =
    'pre-line';

  const form =
    $('uploadForm');

  if (form) {

    const submitButton =
      form.querySelector(
        '#createRunBtn'
      ) ||
      form.querySelector(
        'button[type="submit"]'
      );

    if (submitButton) {
      submitButton.parentNode.insertBefore(
        el,
        submitButton
      );
    } else {
      form.appendChild(
        el
      );
    }
  }

  return el;
}

function setUploadValidationMessage(
  message,
  valid = false
) {
  const el =
    getOrCreateUploadValidation();

  el.textContent =
    message;

  el.style.color =
    valid
      ? 'var(--success, #22c55e)'
      : 'var(--danger, #ef4444)';

  el.style.fontSize =
    '0.9rem';

  el.style.lineHeight =
    '1.5';

  return el;
}

function clearUploadValidation() {
  const el =
    $('uploadValidation');

  if (el) {
    el.textContent =
      '';
  }
}

async function validateUploadSelection() {
  const pdfFiles = [
    ...$('pdfInput').files,
  ];

  const metadataFile =
    $('metadataInput')
      .files[0];

  const errors =
    await validatePdfMetadata(
      pdfFiles,
      metadataFile
    );

  if (errors.length) {

    setUploadValidationMessage(
      `Metadata validation failed:\n${errors
        .map(
          error =>
            `• ${error}`
        )
        .join('\n')}`,
      false
    );

    return false;
  }

  const pdfIds =
    pdfFiles
      .map(
        file =>
          getPdfId(
            file.name
          )
      )
      .sort(
        (a, b) =>
          a - b
      );

  setUploadValidationMessage(
    `Metadata validated successfully.\n` +
      pdfIds
        .map(
          id =>
            `${id}.pdf → metadata meta.id: "${id}"`
        )
        .join('\n'),
    true
  );

  return true;
}


/* ============================================================
   PDF DROPZONE
   ============================================================ */

const pdfDropzone =
  document.querySelector(
    '.dropzone'
  );

function updatePdfFileList() {
  const files =
    [...$('pdfInput').files];

  $('fileList').innerHTML =
    files
      .map(
        file =>
          `
            <div class="file-chip">
              ${escapeHtml(
                file.name
              )}
            </div>
          `
      )
      .join('');
}

function addDroppedPdfFiles(
  files
) {
  const pdfFiles =
    [...files].filter(
      file =>
        file.type ===
          'application/pdf' ||
        file.name
          .toLowerCase()
          .endsWith('.pdf')
    );

  if (!pdfFiles.length) {

    setUploadValidationMessage(
      'Please drop PDF files only.',
      false
    );

    return;
  }

  const dataTransfer =
    new DataTransfer();

  const existingFiles =
    [...$('pdfInput').files];

  const allFiles =
    [
      ...existingFiles,
      ...pdfFiles,
    ];

  const seen =
    new Set();

  for (
    const file of
      allFiles
  ) {

    const key =
      `${file.name}:${file.size}:${file.lastModified}`;

    if (
      seen.has(key)
    ) {
      continue;
    }

    seen.add(key);

    dataTransfer.items.add(
      file
    );
  }

  $('pdfInput').files =
    dataTransfer.files;

  updatePdfFileList();

  validateUploadSelection();
}

if (pdfDropzone) {

  pdfDropzone.addEventListener(
    'dragenter',
    event => {

      event.preventDefault();
      event.stopPropagation();

      pdfDropzone.classList.add(
        'drag-over'
      );
    }
  );

  pdfDropzone.addEventListener(
    'dragover',
    event => {

      event.preventDefault();
      event.stopPropagation();

      pdfDropzone.classList.add(
        'drag-over'
      );

      if (
        event.dataTransfer
      ) {
        event.dataTransfer.dropEffect =
          'copy';
      }
    }
  );

  pdfDropzone.addEventListener(
    'dragleave',
    event => {

      event.preventDefault();
      event.stopPropagation();

      if (
        !pdfDropzone.contains(
          event.relatedTarget
        )
      ) {
        pdfDropzone.classList.remove(
          'drag-over'
        );
      }
    }
  );

  pdfDropzone.addEventListener(
    'drop',
    event => {

      event.preventDefault();
      event.stopPropagation();

      pdfDropzone.classList.remove(
        'drag-over'
      );

      if (
        event.dataTransfer
          ?.files
      ) {
        addDroppedPdfFiles(
          event.dataTransfer.files
        );
      }
    }
  );
}


/*
 * Prevent browser from opening PDFs when
 * dropped outside the dropzone.
 */
document.addEventListener(
  'dragover',
  event => {
    event.preventDefault();
  }
);

document.addEventListener(
  'drop',
  event => {

    if (
      !pdfDropzone ||
      !pdfDropzone.contains(
        event.target
      )
    ) {
      event.preventDefault();
    }
  }
);


/* ============================================================
   STAGE UI
   ============================================================ */

function showStage() {
  $('startView')
    .classList.add(
      'hidden'
    );

  $('stageView')
    .classList.remove(
      'hidden'
    );

  const stage =
    currentStage();

  const info =
    currentStageInfo();

  if (
    !stage ||
    !info
  ) {
    return;
  }

  const artifacts =
    state.run
      ?.artifacts?.[
        stage.id
      ] || [];

  const isComplete =
    info.status ===
    'complete';

  const isRunning =
    info.status ===
    'running';

  const isError =
    info.status ===
    'error';

  const isLastStage =
    state.stageIndex ===
    state.pipeline.length - 1;

  const lastCompletedIndex =
    getLastCompletedIndex();

  const isLatestCompletedStage =
    isComplete &&
    state.stageIndex ===
      lastCompletedIndex;

  const isHistoricalCompletedStage =
    isComplete &&
    state.stageIndex <
      lastCompletedIndex;

  const canProceed =
    isLatestCompletedStage &&
    artifacts.length > 0;


  /* ----------------------------------------------------------
     HEADER
     ---------------------------------------------------------- */

  $('stageKicker')
    .textContent =
      `MODULE ${String(
        stage.number
      ).padStart(
        2,
        '0'
      )}`;

  $('stageTitle')
    .textContent =
    stage.title;

  $('stageDescription')
    .textContent =
    stage.short;

  $('stageNumber')
    .textContent =
      String(
        stage.number
      ).padStart(
        2,
        '0'
      );


  /* ----------------------------------------------------------
     STATUS
     ---------------------------------------------------------- */

  if (
    isComplete
  ) {

    $('stageStateLabel')
      .textContent =
      'Completed';

  } else if (
    isRunning
  ) {

    $('stageStateLabel')
      .textContent =
      'Running';

  } else if (
    isError
  ) {

    $('stageStateLabel')
      .textContent =
      'Failed';

  } else {

    $('stageStateLabel')
      .textContent =
      'Ready to run';
  }


  /* ----------------------------------------------------------
     OUTPUT
     ---------------------------------------------------------- */

  $('stageArtifactLabel')
    .textContent =
      artifacts.length > 0
        ? 'Output saved'
        : 'Waiting for output';


  /* ----------------------------------------------------------
     RUN BUTTON
     ---------------------------------------------------------- */

  $('runBtn')
    .classList.remove(
      'hidden'
    );

  const canRun =
    !isRunning &&
    !isComplete &&
    !isPipelineRunning();

  $('runBtn').disabled =
    !canRun;

  $('runBtn')
    .classList.toggle(
      'completed-disabled',
      isComplete
    );

  if (
    isLastStage &&
    isComplete
  ) {
    $('runBtn')
      .classList.add(
        'hidden'
      );
  }


  /* ----------------------------------------------------------
     NEXT BUTTON
     ---------------------------------------------------------- */

  $('nextBtn')
    .classList.toggle(
      'hidden',
      !canProceed ||
        isLastStage
    );

  $('nextBtn').disabled =
    !canProceed ||
    isLastStage;


  /* ----------------------------------------------------------
     RESET BUTTON
     ---------------------------------------------------------- */

  const showReset =
    isLastStage &&
    isComplete &&
    artifacts.length > 0;

  $('resetBtn')
    .classList.toggle(
      'hidden',
      !showReset
    );

  $('resetBtn').disabled =
    !showReset;


  /* ----------------------------------------------------------
     PROGRESS
     ---------------------------------------------------------- */

  if (
    isComplete
  ) {

    $('stageProgress')
      .style.width =
      '100%';

  } else if (
    isRunning
  ) {

    $('stageProgress')
      .style.width =
      '58%';

  } else {

    $('stageProgress')
      .style.width =
      '0%';
  }


  /* ----------------------------------------------------------
     TIMER
     ---------------------------------------------------------- */

  const savedDuration =
    state.stageDurations[
      stage.id
    ] || 0;

  if (
    isRunning &&
    state.stageStartedAt !==
      null
  ) {

    $('stageTimer')
      .textContent =
      formatDuration(
        performance.now() -
          state.stageStartedAt
      );

  } else {

    $('stageTimer')
      .textContent =
      formatDuration(
        savedDuration
      );
  }


  /* ----------------------------------------------------------
     MESSAGE
     ---------------------------------------------------------- */

  if (
    isComplete
  ) {

    if (
      isLastStage
    ) {

      $('stageLog')
        .textContent =
        `Module finished successfully.\nRDF output has been saved. Your full pipeline time is shown below.`;

    } else if (
      isHistoricalCompletedStage
    ) {

      $('stageLog')
        .textContent =
        `This module has already been completed.\nYou can review its output, but it cannot be rerun. Click the latest completed module in the sidebar to continue.`;

    } else {

      $('stageLog')
        .textContent =
        `Module finished successfully.\nOutput has been saved. Click "Save & next" to continue.`;
    }

  } else if (
    isError
  ) {

    $('stageLog')
      .textContent =
      info.error ||
      'This module failed. Click "Run module" to retry it.';

  } else if (
    isRunning
  ) {

    $('stageLog')
      .textContent =
      `Running ${stage.title}…\nThis can take a while.`;

  } else {

    $('stageLog')
      .textContent =
      'Click "Run module" to start this stage.';
  }


  /* ----------------------------------------------------------
     TOTAL TIME
     ---------------------------------------------------------- */

  if (
    isLastStage &&
    isComplete
  ) {

    $('totalTimeCard')
      .classList.remove(
        'hidden'
      );

    $('totalTimeValue')
      .textContent =
      formatDuration(
        state.totalCompletedMs
      );

  } else {

    $('totalTimeCard')
      .classList.add(
        'hidden'
      );
  }


  /* ----------------------------------------------------------
     VISUALIZATION / STATS / SIDEBAR
     ---------------------------------------------------------- */

  renderPreview(
    stage.id
  );

  renderStats(
    stage.id
  );

  renderSteps();
}


/* ============================================================
   THEME
   ============================================================ */

$('themeToggle')
  .addEventListener(
    'click',
    () => {

      const next =
        document.documentElement
          .dataset
          .theme ===
          'dark'
          ? 'light'
          : 'dark';

      applyTheme(
        next
      );
    }
  );


/* ============================================================
   PDF / METADATA INPUTS
   ============================================================ */

$('pdfInput')
  .addEventListener(
    'change',
    async () => {

      updatePdfFileList();

      await validateUploadSelection();
    }
  );

$('metadataInput')
  .addEventListener(
    'change',
    async () => {

      await validateUploadSelection();
    }
  );


/* ============================================================
   CREATE RUN
   ============================================================ */

$('uploadForm')
  .addEventListener(
    'submit',
    async event => {

      event.preventDefault();

      const pdfFiles =
        [...$('pdfInput').files];

      const metadataFile =
        $('metadataInput')
          .files[0];

      const valid =
        await validateUploadSelection();

      if (!valid) {
        return;
      }

      const btn =
        $('createRunBtn');

      btn.disabled =
        true;

      btn.textContent =
        'Creating run…';

      try {

        const formData =
          new FormData();

        pdfFiles.forEach(
          file => {

            formData.append(
              'pdfs',
              file
            );
          }
        );

        formData.append(
          'metadata',
          metadataFile
        );

        state.run =
          await api(
            '/api/runs',
            {
              method:
                'POST',

              body:
                formData,
            }
          );

        state.stageIndex =
          0;

        state.stageDurations =
          {};

        state.totalCompletedMs =
          0;

        state.stageStartedAt =
          null;

        state.visualization = {
          stageId: null,
          items: [],
          index: 0,
        };

        state.visualizationCache =
          {};

        state.visualizationIndex =
          {};

        showStage();

      } catch (err) {

        alert(
          err.message
        );

      } finally {

        btn.disabled =
          false;

        btn.innerHTML =
          'Create run <span>→</span>';
      }
    }
  );


/* ============================================================
   RUN MODULE
   ============================================================ */

$('runBtn')
  .addEventListener(
    'click',
    async () => {

      const stage =
        currentStage();

      const btn =
        $('runBtn');

      if (!stage) {
        return;
      }

      const info =
        currentStageInfo();

      /*
       * Completed stages cannot be rerun.
       */
      if (
        info?.status ===
        'complete'
      ) {
        return;
      }

      /*
       * Never start a second module while
       * another one is running.
       */
      if (
        isPipelineRunning()
      ) {
        return;
      }

      btn.disabled =
        true;

      state.stageDurations[
        stage.id
      ] = 0;

      startStageTimer(
        stage.id
      );

      /*
       * Immediately update frontend.
       */
      if (
        state.run?.stages?.[
          stage.id
        ]
      ) {

        state.run.stages[
          stage.id
        ].status =
          'running';
      }

      $('stageStateLabel')
        .textContent =
        'Running';

      $('stageProgress')
        .style.width =
        '58%';

      $('stageLog')
        .textContent =
        `Running ${stage.title}…\nThis can take a while.`;

      /*
       * Lock sidebar immediately.
       */
      renderSteps();

      try {

        const result =
          await api(
            `/api/runs/${state.run.run_id}/stages/${stage.id}/run`,
            {
              method:
                'POST',
            }
          );

        finishStageTimer(
          stage.id
        );

        state.run =
          result.run;

        showStage();

      } catch (err) {

        finishStageTimer(
          stage.id
        );

        /*
         * Refresh server state because a long
         * request can finish server-side even
         * when the frontend connection fails.
         */
        try {

          state.run =
            await api(
              `/api/runs/${state.run.run_id}`
            );

        } catch (_) {
          /*
           * Keep current state if refresh fails.
           */
        }

        const latestInfo =
          state.run
            ?.stages?.[
              stage.id
            ];

        if (
          latestInfo?.status ===
          'complete'
        ) {

          showStage();

          return;
        }

        $('stageStateLabel')
          .textContent =
          'Failed';

        $('stageProgress')
          .style.width =
          '0%';

        $('stageLog')
          .textContent =
          `${err.message}\n\nYou can retry this module.`;

        btn.disabled =
          false;

        showStage();
      }
    }
  );


/* ============================================================
   SAVE & NEXT
   ============================================================ */

$('nextBtn')
  .addEventListener(
    'click',
    async () => {

      const stage =
        currentStage();

      const btn =
        $('nextBtn');

      if (!stage) {
        return;
      }

      const lastCompletedIndex =
        getLastCompletedIndex();

      /*
       * Only newest completed stage
       * can continue.
       */
      if (
        state.stageIndex !==
        lastCompletedIndex
      ) {
        return;
      }

      const artifacts =
        state.run
          ?.artifacts?.[
            stage.id
          ] || [];

      if (
        state.run
          ?.stages?.[
            stage.id
          ]?.status !==
          'complete'
      ) {
        return;
      }

      if (
        artifacts.length ===
        0
      ) {
        return;
      }

      if (
        state.stageIndex >=
        state.pipeline.length - 1
      ) {
        return;
      }

      btn.disabled =
        true;

      try {

        const updatedRun =
          await api(
            `/api/runs/${state.run.run_id}/stages/${stage.id}/next`,
            {
              method:
                'POST',

              headers: {
                'content-type':
                  'application/json',
              },

              body:
                JSON.stringify({
                  stage:
                    stage.id,
                }),
            }
          );

        state.run =
          updatedRun;

        state.stageIndex +=
          1;

        state.stageStartedAt =
          null;

        showStage();

      } catch (err) {

        btn.disabled =
          false;

        alert(
          err.message
        );
      }
    }
  );


/* ============================================================
   RESET
   ============================================================ */

$('resetBtn')
  .addEventListener(
    'click',
    () => {

      showStartView();

      window.scrollTo({
        top:
          0,

        behavior:
          'smooth',
      });
    }
  );


/* ============================================================
   INITIALIZATION
   ============================================================ */

(async function init() {
  try {

    initTheme();

    const data =
      await api(
        '/api/pipeline'
      );

    state.pipeline =
      data.stages;

    renderSteps();

    $('stageTimer')
      .textContent =
      '00:00';

    clearUploadValidation();

  } catch (err) {

    console.error(
      err
    );

    alert(
      `Could not initialize the application: ${err.message}`
    );
  }
})();