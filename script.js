(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  };
  const post = (url, body) =>
    api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

  const fileToDataURL = f =>
    new Promise((res, rej) => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.onerror = rej;
      fr.readAsDataURL(f);
    });

  const debounce = (fn, ms) => {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  const setMsg = (el, text, isError = false) => {
    el.textContent = text || '';
    el.classList.toggle('error', isError);
  };

  // ------------------------------------------------------------------
  // Abas
  // ------------------------------------------------------------------
  $$('#main-tabs .tab').forEach(b =>
    b.addEventListener('click', () => {
      $$('#main-tabs .tab').forEach(x => x.classList.toggle('active', x === b));
      $$('.view').forEach(v => v.classList.toggle('active', v.id === b.dataset.view));
    })
  );
  $$('#feature-tabs .tab').forEach(b =>
    b.addEventListener('click', () => {
      $$('#feature-tabs .tab').forEach(x => x.classList.toggle('active', x === b));
      $$('.pane').forEach(p => p.classList.toggle('active', p.id === b.dataset.pane));
    })
  );

  // ------------------------------------------------------------------
  // ESP32
  // ------------------------------------------------------------------
  const portSel = $('#port');
  const espStatus = $('#esp-status');

  function updateEspUI(connected, text) {
    $('#btn-connect').hidden = connected;
    $('#btn-disconnect').hidden = !connected;
    setMsg(espStatus, text || (connected ? 'Conectado' : 'Modo simulação (sem ESP32)'));
  }

  async function loadPorts() {
    try {
      const d = await api('/api/ports');
      portSel.innerHTML = '<option value="">Serial simulada (simulação)</option>' +
        d.ports.map(p => `<option>${p}</option>`).join('');
      if (!d.serial_available) setMsg(espStatus, 'pyserial não instalado. Modo simulação.');
      else updateEspUI(d.connected);
    } catch (e) {
      setMsg(espStatus, e.message, true);
    }
  }

  $('#btn-connect').addEventListener('click', async () => {
    if (!portSel.value) return setMsg(espStatus, 'Rodando em simulação serial.');
    try {
      const d = await post('/api/connect', { port: portSel.value, baud: $('#baud').value });
      updateEspUI(true, d.message);
    } catch (e) {
      setMsg(espStatus, e.message, true);
    }
  });
  $('#btn-disconnect').addEventListener('click', async () => {
    await post('/api/disconnect', {});
    updateEspUI(false);
  });

  // ------------------------------------------------------------------
  // Dataset
  // ------------------------------------------------------------------
  let datasetFiles = [];
  let datasetLoaded = false;

  async function loadDataset() {
    if (datasetLoaded) return;
    try {
      const d = await api('/api/dataset');
      datasetFiles = d.files;
      datasetLoaded = true;
      const sel = $('#dataset-file');
      const msg = $('#dataset-msg');
      if (!d.exists) {
        setMsg(msg, `Pasta não encontrada: ${d.path}. Crie a pasta "dataset" ao lado do app.py e coloque as imagens.`, true);
        sel.innerHTML = '';
      } else if (!d.files.length) {
        setMsg(msg, 'Nenhuma imagem encontrada na pasta do dataset.', true);
        sel.innerHTML = '';
      } else {
        setMsg(msg, '');
        $('#dataset-count').textContent = `(${d.files.length} arquivos)`;
        sel.innerHTML = d.files.map(f => `<option value="${f}">${f}</option>`).join('');
      }
      $$('.slot-dataset').forEach(s => {
        s.innerHTML = '<option value="">— ou escolha do dataset —</option>' +
          d.files.map(f => `<option value="${f}">${f}</option>`).join('');
      });
    } catch (e) {
      setMsg($('#dataset-msg'), e.message, true);
    }
  }

  // ------------------------------------------------------------------
  // Fonte da imagem / classificação
  // ------------------------------------------------------------------
  let currentSource = null;
  let camStream = null;
  const minScore = $('#min-score');

  const currentKind = () => $('input[name="source"]:checked').value;

  function stopCamera() {
    if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
  }

  async function startCamera() {
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: true });
      $('#cam').srcObject = camStream;
      setMsg($('#cam-msg'), '');
    } catch (e) {
      setMsg($('#cam-msg'), 'Não foi possível acessar a câmera: ' + e.message, true);
    }
  }

  async function onSourceChange() {
    const kind = currentKind();
    $$('.src').forEach(s => (s.hidden = s.id !== 'src-' + kind));
    stopCamera();
    currentSource = null;
    if (kind === 'synthetic') {
      currentSource = { kind: 'synthetic', type: $('#synthetic-type').value };
      analyze();
    } else if (kind === 'camera') {
      startCamera();
    } else if (kind === 'dataset') {
      await loadDataset();
      if ($('#dataset-file').value) {
        currentSource = { kind: 'dataset', file: $('#dataset-file').value };
        analyze();
      }
    } else if (kind === 'upload') {
      setMsg($('#analyze-msg'), 'Envie uma imagem de tomate no painel acima.');
      $('#results').hidden = true;
      $('#landing').hidden = false;
    }
  }

  $$('input[name="source"]').forEach(r => r.addEventListener('change', onSourceChange));
  $('#synthetic-type').addEventListener('change', e => {
    currentSource = { kind: 'synthetic', type: e.target.value };
    analyze();
  });
  $('#upload-file').addEventListener('change', async e => {
    const f = e.target.files[0];
    if (!f) return;
    currentSource = { kind: 'upload', image: await fileToDataURL(f) };
    analyze();
  });
  $('#dataset-file').addEventListener('change', e => {
    currentSource = { kind: 'dataset', file: e.target.value };
    analyze();
  });
  $('#btn-capture').addEventListener('click', () => {
    const v = $('#cam');
    if (!v.videoWidth) return setMsg($('#cam-msg'), 'A câmera ainda não está pronta.', true);
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    currentSource = { kind: 'camera', image: c.toDataURL('image/jpeg', 0.92) };
    analyze();
  });

  minScore.addEventListener('input', () => ($('#min-score-val').textContent = minScore.value));
  minScore.addEventListener('input', debounce(() => currentSource && analyze(), 400));

  const metric = (label, value) => `<div class="metric"><small>${label}</small><b>${value}</b></div>`;

  async function analyze() {
    if (!currentSource) return;
    const msg = $('#analyze-msg');
    setMsg(msg, 'Analisando parâmetros de qualidade do tomate...');
    try {
      const r = await post('/api/analyze', { source: currentSource, min_score: +minScore.value, send: true });
      setMsg(msg, '');
      render(r);
    } catch (e) {
      setMsg(msg, e.message, true);
    }
  }

  function render(r) {
    $('#landing').hidden = true;
    $('#results').hidden = false;

    $('#img-original').src = r.images.original;
    $('#img-tomato').src = r.images.tomato_only;
    $('#img-color').src = r.images.color_split;
    $('#img-contours').src = r.images.contours;
    $('#img-texture').src = r.images.texture;
    $('#img-defects').src = r.images.defects;

    const good = r.basket === 'A';
    const box = $('#decision-box');
    box.className = 'result ' + (good ? 'good' : 'bad');
    $('#decision-text').textContent = good ? '🟢 CESTA A (BOM)' : '🔴 CESTA B (RUIM)';
    $('#score').textContent = r.total_score;
    $('#decision-hint').textContent = good
      ? 'O tomate atende aos critérios de qualidade. Pronto para embalar.'
      : 'O tomate não atende aos parâmetros. Acionando o rejeitador.';
    $('#serial-log').textContent = r.serial_log;

    const m = r.metrics;
    $('#m-color').innerHTML =
      metric('Vermelho', m.color.red_pct.toFixed(1) + '%') +
      metric('Amarelo', m.color.yellow_pct.toFixed(1) + '%') +
      metric('Verde', m.color.green_pct.toFixed(1) + '%');
    $('#m-size').innerHTML =
      metric('Área do tomate', Math.round(m.size.area_px) + ' px') +
      metric('Proporção (L/A)', m.size.aspect.toFixed(2));
    $('#m-texture').innerHTML = metric('Índice de rugosidade (LBP)', m.texture.roughness.toFixed(2));
    $('#m-defects').innerHTML = metric('Cobertura de manchas', m.health.spot_pct.toFixed(2) + '%');

    const cards = [
      ['Cor', 'Divisão HSV', m.color.score, 25],
      ['Tamanho', m.size.class, m.size.score, 15],
      ['Textura', m.texture.class, m.texture.score, 15],
      ['Maturação', m.ripeness.class, m.ripeness.score, 20],
      ['Saúde', m.health.class, m.health.score, 20],
      ['Defeitos', m.defects.class, m.defects.score, 5],
    ];
    $('#cards').innerHTML = cards
      .map(([t, v, s, max]) =>
        `<div class="card"><div class="t">${t}</div><div class="v">${v}</div><div class="s">Nota: ${s}/${max}</div></div>`)
      .join('');
  }

  // ------------------------------------------------------------------
  // Comparação
  // ------------------------------------------------------------------
  const slotSource = { a: null, b: null };

  $$('.slot').forEach(slot => {
    const key = slot.dataset.slot;
    const preview = $('.slot-preview', slot);
    const fileInput = $('.slot-file', slot);
    const dsSel = $('.slot-dataset', slot);

    fileInput.addEventListener('change', async () => {
      const f = fileInput.files[0];
      if (!f) return;
      const url = await fileToDataURL(f);
      slotSource[key] = { kind: 'upload', image: url };
      dsSel.value = '';
      preview.src = url;
      preview.hidden = false;
    });
    dsSel.addEventListener('change', () => {
      if (!dsSel.value) return;
      slotSource[key] = { kind: 'dataset', file: dsSel.value };
      fileInput.value = '';
      preview.src = '/api/dataset/image?file=' + encodeURIComponent(dsSel.value);
      preview.hidden = false;
    });
  });

  $('#btn-compare').addEventListener('click', async () => {
    const msg = $('#compare-msg');
    if (!slotSource.a || !slotSource.b) return setMsg(msg, 'Escolha as duas imagens (A e B).', true);
    const btn = $('#btn-compare');
    btn.disabled = true;
    setMsg(msg, 'Isolando os tomates e comparando...');
    try {
      const r = await post('/api/compare', { a: slotSource.a, b: slotSource.b });
      setMsg(msg, '');
      $('#compare-results').hidden = false;
      $('#cut-a').src = r.cutouts.a;
      $('#cut-b').src = r.cutouts.b;
      $('#sim-overall').textContent = r.overall.toFixed(1) + '%';
      $('#sim-verdict').textContent = r.verdict;
      $('#compare-box').className = 'result ' + (r.overall >= 70 ? 'good' : r.overall >= 50 ? '' : 'bad');
      const labels = { color: 'Cor', texture: 'Textura', shape: 'Formato', size: 'Tamanho', defects: 'Defeitos' };
      $('#sim-bars').innerHTML = Object.entries(r.parts)
        .map(([k, v]) =>
          `<div class="bar-row"><span>${labels[k]}</span><div class="bar"><i style="width:${v}%"></i></div><b>${v.toFixed(0)}%</b></div>`)
        .join('');
      const d = r.details;
      $('#sim-details').textContent =
        `Diferença de cor (ΔE): ${d.delta_e} · Área A/B: ${d.area_a}px / ${d.area_b}px · Manchas A/B: ${d.spot_a}% / ${d.spot_b}%`;
    } catch (e) {
      setMsg(msg, e.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  // ------------------------------------------------------------------
  // Inicialização
  // ------------------------------------------------------------------
  loadPorts();
  loadDataset();
  onSourceChange();
})();