"""Auditoria final dos artefatos, sem retreino e sem alterar resultados."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .evaluate import CLASSES, EXPERIMENTS, ROOT, verify_lock, sha256_file

def audit(root):
    checks=[]
    lock=verify_lock(root)
    checks.append('Manifesto, código central, configurações, modelos e registros coincidem com o bloqueio.')
    manifest=pd.read_csv(root/'data/manifest.csv')
    initial=pd.read_csv(root/'data/manifest_before_grouping.csv')
    assert len(manifest)==2527 and manifest.path.is_unique
    assert pd.crosstab(manifest.class_name,manifest.split).equals(pd.crosstab(initial.class_name,initial.split))
    assert manifest.groupby('pixel_sha256').split.nunique().max()==1
    assert manifest.groupby('sha256').split.nunique().max()==1
    checks.append('2.527 imagens; duplicatas exatas agrupadas; estratificação preservada.')
    summary=pd.read_csv(root/'reports/summary.csv').set_index('experiment')
    per=pd.read_csv(root/'reports/per_class.csv')
    assert set(summary.index)==set(EXPERIMENTS) and len(per)==36
    states={}
    for e in EXPERIMENTS:
        folder=root/'runs'/e
        m=json.loads((folder/'metadata.json').read_text())
        metric=json.loads((folder/'metrics.json').read_text())
        pred=pd.read_csv(folder/'predictions.csv')
        history=pd.read_csv(folder/'history.csv')
        assert set(pred.path)==set(manifest[manifest.split=='test'].path) and pred.path.is_unique
        assert len(pred)==380 and len(history)==m['epochs_completed']
        assert m['best_epoch']==int(history.loc[history.val_loss.idxmin(),'epoch'])
        assert pd.Timestamp(metric['evaluated_at_utc'])>=pd.Timestamp(lock['created_at_utc'])>=pd.Timestamp(m['finished_at_utc'])
        assert metric['predictions_sha256']==sha256_file(folder/'predictions.csv')
        np.testing.assert_allclose([accuracy_score(pred.y_true,pred.y_pred),f1_score(pred.y_true,pred.y_pred,labels=range(6),average='macro',zero_division=0)],
                                   [summary.loc[e,'test_accuracy'],summary.loc[e,'test_f1_macro']],atol=1e-12)
        np.testing.assert_array_equal(confusion_matrix(pred.y_true,pred.y_pred,labels=range(6)),metric['confusion_count'])
        assert m['config']['seed']==42 and m['config']['batch_size']==32 and m['config']['learning_rate']==.0001
        assert np.isclose(summary.loc[e,'mean_epoch_seconds']*m['epochs_completed'],m['total_training_seconds'])
        states[e]=m['initial_state_sha256']
    assert states[EXPERIMENTS[0]]==states[EXPERIMENTS[1]]
    assert len({states[e] for e in EXPERIMENTS if e.startswith('cnn')})==1
    checks.append('Seis checkpoints; seleção pré-teste; métricas recalculadas; custos e inicializações conferidos.')
    for name in ['01_analise_exploratoria.ipynb','02_resultados.ipynb']:
        n=nbformat.read(root/'notebooks'/name,as_version=4);nbformat.validate(n)
        cells=[c for c in n.cells if c.cell_type=='code']
        assert cells and all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    checks.append('Dois notebooks executados integralmente, sem saídas de erro.')
    figures=list((root/'reports/figures').glob('fig*.png'))
    assert len(figures)==8
    reader=PdfReader(root/'artigo/artigo_redes_neurais.pdf')
    assert 1<=len(reader.pages)<=10
    text='\n'.join(p.extract_text() for p in reader.pages)
    for name in ['Introdução','Materiais e métodos','Resultados','Discussão','Conclusão','Referências']:
        assert name in text
    checks.append(f'Oito figuras e artigo com {len(reader.pages)} páginas e seções exigidas.')
    team=json.loads((root/'artigo/equipe.json').read_text())
    pending=[]
    if not team.get('github_url'):pending.append('Publicar o repositório e inserir o link real no artigo.')
    if any(a['matricula']=='9999999' for a in team['autores']):pending.append('Substituir as matrículas provisórias 9999999.')
    if not team.get('revisao_da_equipe_concluida'):pending.append('Realizar a revisão e discussão pela equipe antes da submissão.')
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'passed':checks,'submission_pending':pending,
            'selected_experiment':lock['selected_experiment'],'pdf_pages':len(reader.pages)}
    (root/'reports/final_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();audit(a.root.resolve())
