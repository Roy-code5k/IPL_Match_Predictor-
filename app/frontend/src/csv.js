// Keep user-controlled strings as text when a CSV is opened in a spreadsheet.
function csvCell(value) {
  let text = value === null || value === undefined ? '' : String(value)
  if (typeof value === 'string' && /^[\s\x00-\x1f\ufeff]*[=+\-@]|^[\t\r\n]/.test(value)) text = `'${text}`
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

export function toCsv(columns, rows) {
  return [columns.map(csvCell).join(','), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(','))].join('\r\n') + '\r\n'
}
