Attribute VB_Name = "PSM_Helper"
' PSM辅助宏 - 用于在Excel中进行基本的数据准备
' 这个宏帮助准备2009年基期的数据

Sub PreparePSMData()
    Dim ws As Worksheet
    Dim wsNew As Worksheet
    Dim lastRow As Long
    Dim i As Long
    Dim targetRow As Long

    MsgBox "此宏将帮助您准备PSM分析所需的2009年基期数据", vbInformation

    ' 检查是否有活动工作表
    If ActiveSheet Is Nothing Then
        MsgBox "请先打开包含数据的工作表！", vbExclamation
        Exit Sub
    End If

    Set ws = ActiveSheet
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row

    ' 创建新工作表
    Set wsNew = Worksheets.Add(After:=Worksheets(Worksheets.Count))
    wsNew.Name = "PSM数据_2009基期"

    ' 复制标题行
    ws.Rows(1).Copy Destination:=wsNew.Rows(1)

    ' 查找2009年数据并复制
    targetRow = 2
    For i = 2 To lastRow
        ' 假设年份在B列（需要根据实际情况调整）
        If ws.Cells(i, 2).Value = 2009 Then
            ws.Rows(i).Copy Destination:=wsNew.Rows(targetRow)
            targetRow = targetRow + 1
        End If
    Next i

    MsgBox "已完成！共提取 " & targetRow - 2 & " 条2009年的数据" & vbCrLf & _
           "数据已保存在 'PSM数据_2009基期' 工作表中" & vbCrLf & vbCrLf & _
           "注意：完整的PSM分析需要Python环境，请参考安装运行指南", vbInformation
End Sub

Sub CheckRequiredVariables()
    Dim ws As Worksheet
    Dim lastCol As Long
    Dim i As Long
    Dim requiredVars As Variant
    Dim foundVars As String
    Dim missingVars As String

    requiredVars = Array("ln_real_gdp", "In_人口密度", "In_金融发展水平", "第二产业占GDP比重")

    If ActiveSheet Is Nothing Then
        MsgBox "请先打开包含数据的工作表！", vbExclamation
        Exit Sub
    End If

    Set ws = ActiveSheet
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    foundVars = ""
    missingVars = ""

    For Each var In requiredVars
        Dim found As Boolean
        found = False
        For i = 1 To lastCol
            If ws.Cells(1, i).Value = var Then
                found = True
                Exit For
            End If
        Next i

        If found Then
            foundVars = foundVars & "✓ " & var & vbCrLf
        Else
            missingVars = missingVars & "✗ " & var & vbCrLf
        End If
    Next var

    MsgBox "变量检查结果：" & vbCrLf & vbCrLf & _
           "找到的变量：" & vbCrLf & foundVars & vbCrLf & _
           "缺失的变量：" & vbCrLf & missingVars & vbCrLf & _
           vbCrLf & "请确保所有所需的变量都在数据中", vbInformation
End Sub
