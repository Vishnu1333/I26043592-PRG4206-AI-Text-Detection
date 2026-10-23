#!/usr/bin/env python3
"""Generate a PDF summary from saved evaluation artefacts and optional live run log.

Without --runtime-json, the PDF is an artefact summary and explicitly says
that Google Colab execution has not yet been recorded. The Colab notebook
writes a runtime JSON after executing the tests and model inference, then calls
this script again to produce a run-specific output PDF.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', default=None)
    parser.add_argument('--runtime-json', default=None)
    return parser.parse_args()


def fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except (TypeError, ValueError):
        return str(x)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output).resolve() if args.output else root / 'outputs' / 'PRG4206_Project_Outputs_Summary.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = root / 'results' / 'metrics' / 'model_comparison.csv'
    stats_path = root / 'results' / 'metrics' / 'statistical_tests_verified.csv'
    cm_path = root / 'results' / 'metrics' / 'confusion_matrices_verified.csv'
    metrics = pd.read_csv(metrics_path)
    stats = pd.read_csv(stats_path)
    cms = pd.read_csv(cm_path)
    runtime = load_json(args.runtime_json) if args.runtime_json else None

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=19, leading=24, alignment=TA_CENTER, textColor=colors.HexColor('#143A66'), spaceAfter=8))
    styles.add(ParagraphStyle(name='CoverSub', parent=styles['Normal'], fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=colors.HexColor('#46566B'), spaceAfter=6))
    styles.add(ParagraphStyle(name='H1Blue', parent=styles['Heading1'], fontSize=15, leading=18, textColor=colors.HexColor('#164E7A'), spaceBefore=6, spaceAfter=8))
    styles.add(ParagraphStyle(name='H2Blue', parent=styles['Heading2'], fontSize=11.5, leading=14, textColor=colors.HexColor('#1D608C'), spaceBefore=5, spaceAfter=5))
    styles.add(ParagraphStyle(name='BodySmall', parent=styles['BodyText'], fontSize=8.5, leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name='FootSmall', parent=styles['BodyText'], fontSize=7, leading=9, textColor=colors.HexColor('#4B5563')))
    styles.add(ParagraphStyle(name='Banner', parent=styles['BodyText'], fontSize=9, leading=13, borderColor=colors.HexColor('#D08A00'), borderWidth=0.8, borderPadding=7, backColor=colors.HexColor('#FFF7E5'), spaceBefore=6, spaceAfter=10))

    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm, topMargin=17*mm, bottomMargin=18*mm, title='PRG4206 Project Outputs Summary', author='Vishnu Vardhan')
    story = []
    story += [Spacer(1, 12*mm), Paragraph('PRG4206 MACHINE LEARNING', styles['CoverSub']), Paragraph('Project Execution & Evaluation Outputs', styles['CoverTitle']), Paragraph('A Comparative and Explainable Machine Learning System for AI-Generated Text Detection', styles['CoverSub']), Spacer(1, 5*mm)]
    story.append(Paragraph('<b>Student:</b> Vishnu Vardhan &nbsp;&nbsp; <b>Student ID:</b> I26043592<br/><b>Dataset:</b> HC3 — Human ChatGPT Comparison Corpus<br/><b>Models:</b> Logistic Regression, Linear SVM, XGBoost', styles['BodySmall']))
    if runtime:
        doc_checks = runtime.get('document_checks', {})
        doc_ok = doc_checks.get('all_succeeded', True)
        run_status = 'PASS' if runtime.get('tests', {}).get('returncode') == 0 and runtime.get('statistics', {}).get('returncode') == 0 and runtime.get('inference', {}).get('all_models_succeeded', False) and doc_ok else 'REVIEW REQUIRED'
        story.append(Paragraph(f'<b>Colab execution status: {run_status}</b><br/>This report includes the runtime record saved by the accompanying Google Colab notebook. Runtime recorded as: {runtime.get("timestamp_local", runtime.get("timestamp_utc", "not recorded"))}.', styles['BodySmall']))
    else:
        story.append(Paragraph('<b>Important status note:</b> This copy is a summary of the saved benchmark artefacts already included in the project. It is <b>not</b> evidence that Google Colab has been run. Open and execute <i>notebooks/00_PRG4206_Colab_Run_and_Outputs.ipynb</i> to run the checks in Colab; the notebook will regenerate this PDF with the actual run record.', styles['Banner']))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph('1. Model comparison on the held-out test set', styles['H1Blue']))
    story.append(Paragraph('The three saved classifiers were evaluated on the same 11,900-example test partition. Values below are read from the bundled verified evaluation file.', styles['BodySmall']))
    header = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'MCC']
    rows = [header]
    for _, row in metrics.iterrows():
        rows.append([str(row['model']), fmt_pct(row['accuracy']), fmt_pct(row['precision']), fmt_pct(row['recall']), f"{float(row['f1']):.4f}", f"{float(row['roc_auc']):.4f}", f"{float(row['pr_auc']):.4f}", f"{float(row['mcc']):.4f}"])
    tbl = Table(rows, colWidths=[34*mm, 18*mm, 19*mm, 17*mm, 15*mm, 18*mm, 17*mm, 15*mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')), ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),7.2), ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('LEFTPADDING',(0,0),(-1,-1),4), ('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph('Main reading: Linear SVM has the highest observed accuracy and F1-score, while Logistic Regression is very close. The paired significance test below does not establish a statistically significant difference between those two models.', styles['BodySmall']))
    for fname, title in [('model_comparison_accuracy.png','Accuracy comparison'),('model_comparison_f1.png','F1-score comparison'),('model_comparison_roc_auc.png','ROC-AUC comparison')]:
        img_path = root / 'results' / 'figures' / fname
        if img_path.exists():
            img = RLImage(str(img_path), width=145*mm, height=60*mm, kind='proportional')
            story.append(KeepTogether([Paragraph(title, styles['H2Blue']), img]))
    story.append(PageBreak())

    story.append(Paragraph('2. Confusion matrices and paired statistical checks', styles['H1Blue']))
    story.append(Paragraph('Confusion-matrix counts use Human-Written as the negative class and AI-Generated as the positive class.', styles['BodySmall']))
    cm_rows = [['Model','TN','FP','FN','TP','Total']]
    for _, r in cms.iterrows():
        cm_rows.append([str(r['model']), f"{int(r['tn']):,}", f"{int(r['fp']):,}", f"{int(r['fn']):,}", f"{int(r['tp']):,}", f"{int(r['total']):,}"])
    cm_tbl = Table(cm_rows, colWidths=[45*mm, 22*mm, 22*mm, 22*mm, 22*mm, 22*mm], repeatRows=1)
    cm_tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(cm_tbl)
    story.append(Spacer(1, 5*mm))
    stat_rows = [['Model comparison','Both correct','A only correct','B only correct','Both wrong','Corrected χ²','p-value','Conclusion']]
    for _, r in stats.iterrows():
        pval = '<0.000001' if float(r['p_value']) == 0.0 else f"{float(r['p_value']):.4f}"
        is_sig = str(r.get('significant', '')).strip().lower() in ('true', '1', 'yes')
        conclusion = 'Significant' if is_sig else 'Not significant'
        stat_rows.append([str(r['comparison']), f"{int(r['both_correct']):,}", f"{int(r['model_a_only_correct']):,}", f"{int(r['model_b_only_correct']):,}", f"{int(r['both_wrong']):,}", f"{float(r['statistic']):.4f}", pval, conclusion])
    stbl = Table(stat_rows, colWidths=[38*mm,17*mm,19*mm,19*mm,16*mm,16*mm,17*mm,25*mm], repeatRows=1)
    stbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),6.2),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(stbl)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph('Interpretation: SVM and Logistic Regression are not significantly different under the paired McNemar test (p = 0.1443). Both linear models differ significantly from XGBoost on this held-out partition (p < 0.000001). These results apply to this benchmark and split; they do not prove universal authorship detection.', styles['BodySmall']))
    cm_img = root / 'results' / 'figures' / 'linear_svm_confusion_matrix.png'
    if cm_img.exists(): story.append(RLImage(str(cm_img), width=110*mm, height=85*mm, kind='proportional'))
    story.append(PageBreak())

    story.append(Paragraph('3. Domain-level results and explainability', styles['H1Blue']))
    domain_files = {
        'Logistic Regression': root / 'results/error_analysis/logistic_regression_domain_metrics.csv',
        'Linear SVM': root / 'results/error_analysis/linear_svm_domain_metrics.csv',
        'XGBoost': root / 'results/error_analysis/xgboost_domain_metrics.csv',
    }
    domain_data = {}
    for model, path in domain_files.items():
        if path.exists():
            d = pd.read_csv(path).set_index('domain')
            domain_data[model] = d
    if domain_data:
        order = ['medicine','finance','reddit_eli5','open_qa','wiki_csai']
        dr = [['Domain','Test samples','LR F1','SVM F1','XGBoost F1']]
        for domain in order:
            if domain in domain_data['Linear SVM'].index:
                count = int(domain_data['Linear SVM'].loc[domain,'n_samples'])
                dr.append([domain, f'{count:,}', f"{domain_data['Logistic Regression'].loc[domain,'f1']:.4f}", f"{domain_data['Linear SVM'].loc[domain,'f1']:.4f}", f"{domain_data['XGBoost'].loc[domain,'f1']:.4f}"])
        dt = Table(dr, colWidths=[39*mm,28*mm,28*mm,28*mm,32*mm], repeatRows=1)
        dt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(dt)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph('The domain results are based on test-set samples, not the full domain totals. Performance is weakest in wiki_csai for these models, which reinforces the need for domain-specific evaluation.', styles['BodySmall']))
    for fname, title, w, h in [
        ('linear_svm_domain_f1.png','Linear SVM F1 by domain',140*mm,65*mm),
        ('xgb_shap_summary.png','SHAP summary for XGBoost',140*mm,70*mm),
    ]:
        p = root / 'results/figures' / fname
        if p.exists():
            story += [Paragraph(title, styles['H2Blue']), RLImage(str(p), width=w, height=h, kind='proportional'), Spacer(1,3*mm)]
    story.append(Paragraph('Top saved XGBoost stylometric features include average word length (22.44%), paragraph count (11.67%), unique-word ratio (11.52%), type-token ratio (6.40%), and character count (5.44%). Feature importance describes behaviour of this trained model; it is not a universal rule for detecting AI authorship.', styles['BodySmall']))
    story.append(PageBreak())

    story.append(Paragraph('4. Colab runtime and demonstration outputs', styles['H1Blue']))
    if runtime:
        env = runtime.get('environment', {})
        info = [
            ['Runtime item','Recorded value'],
            ['Timestamp', str(runtime.get('timestamp_local', runtime.get('timestamp_utc','Not recorded')))],
            ['Python', str(env.get('python','Not recorded'))],
            ['Platform', str(env.get('platform','Not recorded'))],
            ['Project root', str(runtime.get('project_root','Not recorded'))],
            ['Pytest', str(runtime.get('tests',{}).get('summary','Not recorded'))],
            ['Statistical verification', str(runtime.get('statistics',{}).get('summary','Not recorded'))],
            ['All three sample predictions successful', str(runtime.get('inference',{}).get('all_models_succeeded',False))],
        ]
        it = Table(info, colWidths=[52*mm,130*mm], repeatRows=1)
        it.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(it)
        story.append(Spacer(1, 4*mm))
        preds = runtime.get('inference',{}).get('predictions',[])
        if preds:
            pr = [['Input case','Model','Prediction','Confidence']]
            for item in preds:
                pr.append([str(item.get('case',''))[:38],str(item.get('model','')),str(item.get('label','')), f"{float(item.get('confidence',0)):.4f}"])
            pt = Table(pr, colWidths=[48*mm,45*mm,48*mm,28*mm], repeatRows=1)
            pt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7.5),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.append(Paragraph('Live text-inference demonstrations (the outputs below are recorded from the actual notebook run; they are examples, not benchmark scores):', styles['BodySmall']))
            story.append(pt)
        document_items = runtime.get('document_checks', {}).get('items', [])
        if document_items:
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph('Sample document parser checks', styles['H2Blue']))
            drows = [['File', 'Format', 'Words', 'Chunks', 'Aggregate label', 'Status']]
            for item in document_items:
                drows.append([
                    str(item.get('filename',''))[:32],
                    str(item.get('format','')),
                    str(item.get('word_count','—')),
                    str(item.get('chunks','—')),
                    str(item.get('prediction','—')),
                    'PASS' if item.get('success') else 'FAILED'
                ])
            dtable = Table(drows, colWidths=[52*mm,17*mm,18*mm,18*mm,45*mm,20*mm], repeatRows=1)
            dtable.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164E7A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#AEBAC6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6FA')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.append(dtable)
            failed_docs = [x for x in document_items if not x.get('success')]
            if failed_docs:
                story.append(Paragraph('Document parser errors: ' + '; '.join(str(x.get('filename')) + ': ' + str(x.get('error','unknown error')) for x in failed_docs), styles['FootSmall']))
        story.append(Spacer(1, 4*mm))
        # Include concise real command outputs rather than risking an oversized appendix.
        for key, title in [('tests','Pytest output excerpt'),('statistics','Statistical verification output excerpt')]:
            section = runtime.get(key,{})
            output = (section.get('stdout','') + '\n' + section.get('stderr','')).strip()
            if output:
                output = output[-2200:]
                output = output.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('\n','<br/>')
                story.append(Paragraph(title, styles['H2Blue']))
                story.append(Paragraph(output, styles['FootSmall']))
    else:
        story.append(Paragraph('The accompanying Colab notebook is the execution step. It uploads this code package, installs the lightweight runtime requirements, loads the three saved models, runs the automated tests and statistical verification, executes demonstration predictions, and regenerates this PDF with the actual runtime record. The current packaged PDF is not a substitute for running that notebook.', styles['Banner']))
        story.append(Paragraph('Suggested Colab workflow', styles['H2Blue']))
        for item in [
            '1. Open the notebook with Google Colab and run cells from top to bottom.',
            '2. When prompted, upload I26043592_Vishnu_Vardhan_Code.zip from your Downloads folder.',
            '3. Review the test summary, independent statistical verification, three-model inference table, and document-parser checks.',
            '4. Download the regenerated outputs PDF and updated ZIP from the notebook final cell.',
        ]:
            story.append(Paragraph(item, styles['BodySmall']))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph('Interpretation and limitations', styles['H1Blue']))
    story.append(Paragraph('The classifier estimates whether text resembles the HC3 human-written or ChatGPT-generated examples. High benchmark scores do not prove authorship or guarantee performance on different generators, domains, editing styles, or very short text. Predictions should be treated as a research aid and interpreted with human review.', styles['BodySmall']))
    story.append(Paragraph('Source artefacts: results/metrics/model_comparison.csv; statistical_tests_verified.csv; confusion_matrices_verified.csv; results/error_analysis/*_domain_metrics.csv; figures under results/figures/. All values in this report should be kept aligned with the executable project.', styles['FootSmall']))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#D4DDE6'))
        canvas.line(17*mm, 14*mm, A4[0]-17*mm, 14*mm)
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#536477'))
        canvas.drawString(17*mm, 9*mm, 'PRG4206 | I26043592 | AI-Generated Text Detection')
        canvas.drawRightString(A4[0]-17*mm, 9*mm, f'Page {doc_.page}')
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Created: {out}')
    print(f'Size: {out.stat().st_size:,} bytes')
    print(f'Pages: generated by ReportLab; runtime record present={runtime is not None}')


if __name__ == '__main__':
    main()
