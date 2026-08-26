{{- define "rwkv-statepool.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "rwkv-statepool.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "rwkv-statepool.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "rwkv-statepool.labels" -}}
app.kubernetes.io/name: {{ include "rwkv-statepool.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
