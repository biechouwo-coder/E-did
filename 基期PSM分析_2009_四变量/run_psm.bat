@echo off
cd /d "c:\Users\HP\Desktop\did-CEADs\基期PSM分析_2009_四变量"
python psm_analysis.py > psm_output.txt 2>&1
echo PSM分析完成！结果已保存到 psm_output.txt
type psm_output.txt
pause
