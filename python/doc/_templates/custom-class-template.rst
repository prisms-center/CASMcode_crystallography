{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
  :show-inheritance:

  {% block methods %}
  {% if methods %}
  .. rubric:: {{ _('Methods') }}

  .. autosummary::
    :nosignatures:
    :toctree:
    :template: custom-function-template.rst
  {% for item in methods %}
    {%- if not item.startswith('_') %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block attributes %}
  {% if attributes %}
  .. rubric:: {{ _('Attributes') }}

  .. autosummary::
    :nosignatures:
    :toctree:
    :template: custom-attr-template.rst
    {% for item in attributes %}
      ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
